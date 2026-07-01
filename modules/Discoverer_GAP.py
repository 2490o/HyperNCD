# _*_ coding: utf-8 _*_
"""
Discoverer_GAP.py
Full geometry-aware prototype + hypergraph reasoning version.

Generated as an ablation-friendly variant of modules/Discoverer_hyper.py.
The original Discoverer_hyper.py is not modified.
"""
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import jaccard_index
from torch_scatter import scatter

from modules.Discoverer_hyper import Discoverer as BaseDiscoverer
from modules.Discoverer_hyper import split_tensor_by_list, cosine_similarity



def _pooling_according_label_device(pred_array, plabels_array):
    device = pred_array.device
    plabels_array = torch.as_tensor(plabels_array, dtype=torch.long, device=device)
    uniques = torch.unique(plabels_array, sorted=True)
    uniques = uniques[uniques != -1]
    if uniques.numel() == 0:
        return pred_array.new_empty((0, pred_array.shape[1])), uniques
    max_index = uniques.max() + 1
    scatter_labels = plabels_array.clone()
    scatter_labels[scatter_labels == -1] = max_index
    out = scatter(pred_array, scatter_labels, dim=0, reduce="mean")
    return out[uniques, :], uniques


class Discoverer(BaseDiscoverer):
    """Geometry-aware hypergraph variant; keeps the public class name unchanged."""

    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.prototype_momentum = float(getattr(self.hparams, "prototype_momentum", 0.9))
        self.register_buffer("prototype_memory", torch.empty(0, self.model.feat_dim), persistent=False)
        self.register_buffer("prototype_geo_memory", torch.empty(0, 6), persistent=False)

    def _smooth_one_hot(self, labels, num_classes, device):
        targets = F.one_hot(labels.to(torch.long), num_classes=num_classes).float().to(device)
        smoothing = float(getattr(self.hparams, "label_smoothing", 0.15))
        if smoothing > 0:
            targets = targets * (1.0 - smoothing) + smoothing / num_classes
        return targets

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._epoch_loss_sum = 0.0
        self._epoch_point_loss_sum = 0.0
        self._epoch_region_loss_sum = 0.0
        self._epoch_hyperedge_sum = 0.0
        self._epoch_step_count = 0

    def _compute_region_geometric_cues(self, pts_coords, pts_labels, eps=1e-6):
        """
        Args:
            pts_coords: Tensor [N, 3], point coordinates for one point-cloud view.
            pts_labels: array-like [N], superpoint ids; -1 denotes invalid region.
        Returns:
            geometry_cue: Tensor [R, 6] = [linearity, planarity, scattering, weighted normalized center xyz].
            region_ids: Tensor [R], valid superpoint ids aligned with geometry_cue rows.
        """
        device = pts_coords.device
        pts_labels = torch.as_tensor(pts_labels, dtype=torch.long, device=device)
        valid_mask = pts_labels != -1
        if pts_coords.numel() == 0 or valid_mask.sum() == 0:
            return pts_coords.new_empty((0, 6)), torch.empty(0, dtype=torch.long, device=device)

        region_ids = torch.unique(pts_labels[valid_mask], sorted=True)
        valid_coords = pts_coords[valid_mask]
        coord_mean = valid_coords.mean(dim=0, keepdim=True)
        coord_std = valid_coords.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        coords_norm = (pts_coords - coord_mean) / coord_std

        cues_list, centers_list = [], []
        for rid in region_ids:
            pts = coords_norm[pts_labels == rid]
            if pts.shape[0] == 0:
                centers_list.append(pts_coords.new_zeros(3))
                cues_list.append(pts_coords.new_zeros(3))
                continue

            center = pts.mean(dim=0)
            centers_list.append(center)
            if pts.shape[0] < 3:
                cues_list.append(pts_coords.new_zeros(3))
                continue

            pts_centered = pts - center
            cov = torch.matmul(pts_centered.t(), pts_centered) / max(pts.shape[0] - 1, 1)
            try:
                eigvals = torch.linalg.eigvalsh(cov)
                eigvals = torch.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
                eigvals = torch.flip(eigvals, dims=[0])
                l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]
                cues = torch.stack([(l1 - l2) / (l1 + eps), (l2 - l3) / (l1 + eps), l3 / (l1 + eps)])
                cues = torch.nan_to_num(cues, nan=0.0, posinf=0.0, neginf=0.0)
            except RuntimeError:
                cues = pts_coords.new_zeros(3)
            cues_list.append(cues)

        curvature = torch.stack(cues_list, dim=0)
        centers = torch.stack(centers_list, dim=0)
        center_weight = float(getattr(self.hparams, "geo_center_weight", 0.1))
        geometry_cue = torch.cat([curvature, center_weight * centers], dim=1)
        return torch.nan_to_num(geometry_cue, nan=0.0, posinf=0.0, neginf=0.0), region_ids

    def _build_region_features(self, point_feats, point_coords, superpoint_labels, batch_size):
        """
        Args:
            point_feats: Tensor [N, C], point-level semantic features from out["feats"].
            point_coords: Tensor [N, 3], coordinates without batch index.
            superpoint_labels: list length B, each item contains Ni superpoint ids.
            batch_size: int.
        Returns:
            region_feat: Tensor [R, C], mean-pooled region semantic prototypes.
            region_geo: Tensor [R, 6], geometry cue; zeros when --disable_gap is enabled.
            region_split_sizes: list[int], number of valid regions per point cloud.
            region_ids_per_pcd: list[Tensor], valid superpoint ids aligned with region rows.
        """
        split_sizes = [len(superpoint_labels[i]) for i in range(batch_size)]
        feat_list = split_tensor_by_list(point_feats, split_sizes)
        coord_list = split_tensor_by_list(point_coords, split_sizes)
        disable_gap = bool(getattr(self.hparams, "disable_gap", False))

        region_feat_list, region_geo_list = [], []
        region_split_sizes, region_ids_per_pcd = [], []
        for i in range(batch_size):
            sem_feat, region_ids = _pooling_according_label_device(feat_list[i], superpoint_labels[i])
            region_split_sizes.append(sem_feat.shape[0])
            region_ids_per_pcd.append(region_ids)
            if sem_feat.shape[0] == 0:
                continue
            region_feat_list.append(sem_feat)
            if disable_gap:
                geo = sem_feat.new_zeros((sem_feat.shape[0], 6))
            else:
                geo, geo_ids = self._compute_region_geometric_cues(coord_list[i], superpoint_labels[i])
                if geo.shape[0] != sem_feat.shape[0] or not torch.equal(geo_ids, region_ids):
                    aligned_geo = sem_feat.new_zeros((sem_feat.shape[0], 6))
                    for j, rid in enumerate(region_ids):
                        matched = torch.nonzero(geo_ids == rid, as_tuple=False).flatten()
                        if matched.numel() > 0:
                            aligned_geo[j] = geo[matched[0]]
                    geo = aligned_geo
            region_geo_list.append(geo)

        if region_feat_list:
            region_feat = torch.cat(region_feat_list, dim=0)
            region_geo = torch.cat(region_geo_list, dim=0)
        else:
            region_feat = point_feats.new_empty((0, point_feats.shape[1]))
            region_geo = point_feats.new_empty((0, 6))
        return region_feat, region_geo, region_split_sizes, region_ids_per_pcd

    def _update_prototype_memory(self, region_feat, region_geo):
        if region_feat.shape[0] == 0:
            return
        momentum = float(getattr(self.hparams, "prototype_momentum", self.prototype_momentum))
        momentum = min(max(momentum, 0.0), 0.999)
        cur_feat, cur_geo = region_feat.detach(), region_geo.detach()
        if self.prototype_memory.numel() == 0:
            self.prototype_memory = cur_feat.clone()
            self.prototype_geo_memory = cur_geo.clone()
            return
        mem_feat = self.prototype_memory.to(cur_feat.device)
        mem_geo = self.prototype_geo_memory.to(cur_geo.device)
        keep = min(mem_feat.shape[0], cur_feat.shape[0])
        new_feat, new_geo = cur_feat.clone(), cur_geo.clone()
        if keep > 0:
            new_feat[:keep] = momentum * mem_feat[:keep] + (1.0 - momentum) * cur_feat[:keep]
            new_geo[:keep] = momentum * mem_geo[:keep] + (1.0 - momentum) * cur_geo[:keep]
        self.prototype_memory = new_feat.detach()
        self.prototype_geo_memory = new_geo.detach()

    def _apply_geometry_aware_hypergraph(self, region_feat, region_geo, return_indices=False):
        """
        Args:
            region_feat: Tensor [R, C], region semantic prototypes.
            region_geo: Tensor [R, 6], geometric cues.
            return_indices: bool, returns Tensor [R, K] top-k neighbor ids if True.
        Returns:
            region_feat_hyper: Tensor [R, C], residual hypergraph-aggregated features.
            hyperedge_weight: scalar Tensor, mean selected top-k similarity.
            topk_indices: Tensor [R, K] or None.
        """
        device = region_feat.device
        num_nodes = region_feat.shape[0]
        if num_nodes == 0:
            empty_idx = torch.empty(0, 0, dtype=torch.long, device=device)
            return region_feat, region_feat.new_tensor(0.0), empty_idx if return_indices else None

        if bool(getattr(self.hparams, "disable_hypergraph", False)) or num_nodes <= 1:
            idx = torch.arange(num_nodes, device=device).view(num_nodes, 1)
            return F.normalize(region_feat, p=2, dim=1), region_feat.new_tensor(1.0), idx if return_indices else None

        alpha = float(getattr(self.hparams, "hyper_alpha", 0.25))
        tau = max(float(getattr(self.hparams, "hyper_tau", 1.0)), 1e-6)
        topk = int(getattr(self.hparams, "hyper_topk", 8))
        residual = min(max(float(getattr(self.hparams, "hyper_residual", 0.2)), 0.0), 1.0)
        use_memory = bool(getattr(self.hparams, "use_prototype_memory", False))

        if use_memory and self.prototype_memory.numel() > 0 and self.prototype_geo_memory.numel() > 0:
            graph_feat = torch.cat([region_feat, self.prototype_memory.to(device)], dim=0)
            graph_geo = torch.cat([region_geo, self.prototype_geo_memory.to(device)], dim=0)
        else:
            graph_feat, graph_geo = region_feat, region_geo

        graph_feat = torch.nan_to_num(graph_feat, nan=0.0, posinf=0.0, neginf=0.0)
        graph_geo = torch.nan_to_num(graph_geo, nan=0.0, posinf=0.0, neginf=0.0)
        graph_nodes = graph_feat.shape[0]

        feat_norm = F.normalize(graph_feat, p=2, dim=1)
        sem_sim = torch.mm(feat_norm[:num_nodes], feat_norm.t()).clamp(min=0.0)
        if bool(getattr(self.hparams, "disable_gap", False)):
            fused_sim = sem_sim
        else:
            geo_norm = F.normalize(graph_geo, p=2, dim=1)
            dist_sq = torch.cdist(geo_norm[:num_nodes], geo_norm, p=2).pow(2)
            geo_sim = torch.exp(-dist_sq / tau)
            fused_sim = alpha * geo_sim + (1.0 - alpha) * sem_sim
        fused_sim[torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)] = 1.0

        k = min(max(topk, 1), graph_nodes)
        topk_vals, topk_indices = torch.topk(fused_sim, k=k, dim=1)
        weights = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-6)
        aggregated_feat = torch.sum(weights.unsqueeze(-1) * graph_feat[topk_indices], dim=1)
        region_feat_hyper = F.normalize((1.0 - residual) * region_feat + residual * aggregated_feat, p=2, dim=1)
        region_feat_hyper = torch.nan_to_num(region_feat_hyper, nan=0.0, posinf=0.0, neginf=0.0)

        if use_memory:
            self._update_prototype_memory(region_feat, region_geo)
        return region_feat_hyper, topk_vals.mean(), topk_indices if return_indices else None

    def _compute_point_pseudo_labels(self, logits, logits1, mapped_labels, mapped_labels1, mask_lab, mask_lab1, nlc):
        """
        Args:
            logits/logits1: Tensor [N, C] and [N1, C], concatenated known+novel logits.
            mapped_labels/mapped_labels1: Tensor [N] and [N1].
            mask_lab/mask_lab1: Bool Tensor, True for known-class points.
            nlc: int, number of known classes.
        Returns:
            targets/targets1: Tensor with one-hot known labels and SK novel pseudo labels.
            freq: Tensor [num_unlabeled], pseudo-label frequency.
            w_ce/w_reg: scalar Tensors from SemiSinkhornKnopp.
        """
        targets = torch.zeros_like(logits)
        targets1 = torch.zeros_like(logits1)
        if mask_lab.any():
            targets[mask_lab, :nlc] = self._smooth_one_hot(mapped_labels[mask_lab], nlc, logits.device).type_as(targets)
        if mask_lab1.any():
            targets1[mask_lab1, :nlc] = self._smooth_one_hot(mapped_labels1[mask_lab1], nlc, logits1.device).type_as(targets1)

        freq = torch.empty(0, device=logits.device)
        w_ce = torch.tensor(0.0, device=logits.device)
        w_reg = torch.tensor(0.0, device=logits.device)
        if targets[~mask_lab].shape[0] != 0 and targets1[~mask_lab1].shape[0] != 0:
            pseudolabel, w_ce, w_reg = self.cos_sk(logits[~mask_lab, nlc:])
            targets[~mask_lab, nlc:] = pseudolabel.detach().type_as(targets)
            targets1[~mask_lab1, nlc:] = self.cos_sk(logits1[~mask_lab1, nlc:])[0].detach().type_as(targets1)
            freq = torch.bincount(torch.argmax(pseudolabel, dim=1), minlength=self.hparams.num_unlabeled_classes)
            freq = freq.float() / max(pseudolabel.shape[0], 1)
        return targets, targets1, freq, w_ce, w_reg

    def _compute_region_pseudo_labels(self, region_feat, region_geo):
        """
        Args:
            region_feat: Tensor [R, C], semantic region prototypes.
            region_geo: Tensor [R, 6], geometry cue.
        Returns:
            region_logits: Tensor [R, K], cosine similarities to novel prototypes.
            targets_region: Tensor [R, K], SK pseudo labels.
            re_w_ce/re_w_reg: scalar Tensors from region-level SemiSinkhornKnopp.
            hyperedge_weight: scalar Tensor.
        """
        if region_feat.shape[0] == 0:
            empty = region_feat.new_empty((0, self.hparams.num_unlabeled_classes))
            zero = region_feat.new_tensor(0.0)
            return empty, empty, zero, zero, zero
        region_feat_hyper, hyperedge_weight, _ = self._apply_geometry_aware_hypergraph(region_feat, region_geo)
        region_logits = cosine_similarity(region_feat_hyper, self.model.head_unlab.prototypes.kernel.data)
        targets_region, re_w_ce, re_w_reg = self.region_sk(region_logits)
        return region_logits, targets_region.detach().type_as(region_logits), re_w_ce, re_w_reg, hyperedge_weight

    def training_step(self, data, _):
        nlc = self.hparams.num_labeled_classes
        (
            coords, feats, real_labels, selected_idx, mapped_labels, superpoint_lab,
            selected_region_idx, coords1, feats1, _, selected_idx1, mapped_labels1,
            superpoint_lab1, selected_region_idx1, pcd_indexes
        ) = data

        batch_num = pcd_indexes.shape[0]
        pcd_masks = [coords[:, 0] == i for i in range(batch_num)]
        pcd_masks1 = [coords1[:, 0] == i for i in range(batch_num)]
        coords = coords.int()
        coords1 = coords1.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)
        sp_tensor1 = ME.SparseTensor(features=feats1.float(), coordinates=coords1)
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        out1 = self.model(sp_tensor1)
        logits = torch.cat([out["logits_lab"], out["logits_unlab"]], dim=-1)
        logits1 = torch.cat([out1["logits_lab"], out1["logits_unlab"]], dim=-1)
        mask_lab = mapped_labels != self.unknown_label
        mask_lab1 = mapped_labels1 != self.unknown_label

        targets, targets1, freq, w_ce, w_reg = self._compute_point_pseudo_labels(
            logits, logits1, mapped_labels, mapped_labels1, mask_lab, mask_lab1, nlc
        )

        loss_cluster = self.loss(10 * logits, targets1, selected_idx, selected_idx1, pcd_masks, pcd_masks1)
        loss_cluster += self.loss(10 * logits1, targets, selected_idx1, selected_idx, pcd_masks1, pcd_masks)

        all_sp = torch.from_numpy(np.concatenate(superpoint_lab, 0)).int()
        all_sp1 = torch.from_numpy(np.concatenate(superpoint_lab1, 0)).int()
        no_region = (
            torch.all(all_sp == -1) or torch.all(all_sp1 == -1)
            or targets[~mask_lab].shape[0] == 0 or targets1[~mask_lab1].shape[0] == 0
            or self.hparams.alpha == 0
        )

        zero = torch.tensor(0.0, device=self.device)
        re_w_ce, re_w_reg = zero, zero
        loss_cluster_region = zero
        hyperedge_weight, hyperedge_weight1 = zero, zero
        region_w_dict = {}

        if not no_region:
            region_feat, region_geo, region_split_sizes, _ = self._build_region_features(
                out["feats"], coords.float()[:, 1:], superpoint_lab, batch_num
            )
            region_feat1, region_geo1, region_split_sizes1, _ = self._build_region_features(
                out1["feats"], coords1.float()[:, 1:], superpoint_lab1, batch_num
            )
            if region_feat.shape[0] > 0 and region_feat1.shape[0] > 0:
                region_logits, targets_region, re_w_ce, re_w_reg, hyperedge_weight = self._compute_region_pseudo_labels(
                    region_feat, region_geo
                )
                region_logits1, targets_region1, _, _, hyperedge_weight1 = self._compute_region_pseudo_labels(
                    region_feat1, region_geo1
                )
                region_logits_list = split_tensor_by_list(region_logits, region_split_sizes)
                region_logits1_list = split_tensor_by_list(region_logits1, region_split_sizes1)
                region_targets_list = split_tensor_by_list(targets_region, region_split_sizes)
                region_targets1_list = split_tensor_by_list(targets_region1, region_split_sizes1)
                loss_cluster_region = self.region_loss(
                    region_logits_list, region_targets1_list, selected_region_idx, selected_region_idx1, batch_num
                )
                loss_cluster_region += self.region_loss(
                    region_logits1_list, region_targets_list, selected_region_idx1, selected_region_idx, batch_num
                )
                region_w_dict = {f"w/region_w{i}": item.item() for i, item in enumerate(self.region_sk.w[0])}

        loss = loss_cluster + self.hparams.alpha * loss_cluster_region
        w_dict = {f"w/w{i}": item.item() for i, item in enumerate(self.cos_sk.w[0])}
        freq_dict = {f"w/freq{i}": item.item() for i, item in enumerate(freq)} if freq.numel() > 0 else {}
        results = {
            "train/loss": loss.detach(),
            "train/loss_cluster": loss_cluster.detach(),
            "train/loss_cluster_region": loss_cluster_region.detach(),
            "train/loss_w_ce": w_ce.detach().float(),
            "train/loss_w_reg": w_reg.detach().float(),
            "gamma": torch.as_tensor(float(self.cos_sk.gamma), device=self.device),
            "train/re_loss_w_ce": re_w_ce.detach().float(),
            "train/re_loss_w_reg": re_w_reg.detach().float(),
            "regamma": torch.as_tensor(float(self.region_sk.gamma), device=self.device),
            "train/hyperedge_weight": hyperedge_weight.detach(),
            "train/hyperedge_weight_view1": hyperedge_weight1.detach(),
            "train/hyper_alpha": torch.as_tensor(float(getattr(self.hparams, "hyper_alpha", 0.25)), device=self.device),
            "train/hyper_topk": torch.as_tensor(float(getattr(self.hparams, "hyper_topk", 8)), device=self.device),
            "train/hyper_residual": torch.as_tensor(float(getattr(self.hparams, "hyper_residual", 0.2)), device=self.device),
            "train/geo_center_weight": torch.as_tensor(float(getattr(self.hparams, "geo_center_weight", 0.1)), device=self.device),
            "train/use_prototype_memory": torch.as_tensor(float(getattr(self.hparams, "use_prototype_memory", False)), device=self.device),
        }
        results.update(w_dict)
        results.update(freq_dict)
        results.update(region_w_dict)

        if (~mask_lab).any():
            pred_unlab = torch.max(
                F.softmax(out["logits_unlab"][~mask_lab].detach(), dim=1), dim=1
            )[1]
            self.train_ps_cosine.append(pred_unlab.cpu())
            self.train_gt.append(real_labels[~mask_lab].detach().cpu())

        self._epoch_loss_sum += float(loss.detach().cpu())
        self._epoch_point_loss_sum += float(loss_cluster.detach().cpu())
        self._epoch_region_loss_sum += float(loss_cluster_region.detach().cpu())
        self._epoch_hyperedge_sum += float(hyperedge_weight.detach().cpu())
        self._epoch_step_count += 1

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_num)

        self.kl_loss = float(w_reg.detach().cpu())
        re_w_reg_value = float(re_w_reg.detach().cpu()) if torch.is_tensor(re_w_reg) else float(re_w_reg)
        if re_w_reg_value <= self.hparams.ak_bound:
            self.region_count += 1
            if self.region_count == self.hparams.smooth_bound:
                print("Update gamma from {} to {}".format(self.region_sk.gamma, self.region_sk.gamma * self.hparams.gamma_decrease))
                self.region_sk.gamma = self.region_sk.gamma * self.hparams.gamma_decrease
                self.region_count = 0
        else:
            self.region_count = 0

        if self.kl_loss <= self.hparams.ak_bound:
            self.count += 1
            if self.count == self.hparams.smooth_bound:
                print("Update gamma from {} to {}".format(self.cos_sk.gamma, self.cos_sk.gamma * self.hparams.gamma_decrease))
                self.cos_sk.gamma = self.cos_sk.gamma * self.hparams.gamma_decrease
                self.count = 0
        else:
            self.count = 0
        return loss

    def on_train_epoch_end(self):
        """Print one compact epoch summary and clear bookkeeping to avoid memory growth."""
        steps = max(getattr(self, "_epoch_step_count", 0), 1)
        epoch_idx = int(self.current_epoch) + 1
        max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", None)
        max_epochs_text = str(max_epochs) if max_epochs is not None else "?"
        avg_loss = getattr(self, "_epoch_loss_sum", 0.0) / steps
        avg_point_loss = getattr(self, "_epoch_point_loss_sum", 0.0) / steps
        avg_region_loss = getattr(self, "_epoch_region_loss_sum", 0.0) / steps
        avg_hyperedge = getattr(self, "_epoch_hyperedge_sum", 0.0) / steps
        if self.trainer is None or self.trainer.is_global_zero:
            print(
                f"Epoch {epoch_idx}/{max_epochs_text} | "
                f"loss={avg_loss:.4f} | point={avg_point_loss:.4f} | "
                f"region={avg_region_loss:.4f} | hyper={avg_hyperedge:.4f} | "
                f"gamma={float(self.cos_sk.gamma):.4f} | regamma={float(self.region_sk.gamma):.4f}",
                flush=True,
            )
        self.train_ps_cosine = []
        self.train_gt = []
        self.asa = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            idx_logits: torch.Tensor,
            idx_targets: torch.Tensor,
            pcd_mask_logits,
            pcd_mask_targets,
            mask_lab_logits=None,
            mask_lab_targets=None,
    ):
        """Point-level swapped prediction loss with epoch-only logging."""
        loss = logits.new_tensor(0.0)
        valid_pcd = 0
        for pcd in range(len(pcd_mask_logits)):
            _idx_logits = pcd_mask_logits[pcd]
            _idx_targets = pcd_mask_targets[pcd]
            if mask_lab_logits is not None and mask_lab_targets is not None:
                _idx_logits = _idx_logits & ~mask_lab_logits
                _idx_targets = _idx_targets & ~mask_lab_targets

            pcd_logits = logits[_idx_logits]
            pcd_targets = targets[_idx_targets]
            logit_shape = pcd_logits.shape[0]
            target_shape = pcd_targets.shape[0]
            if logit_shape == 0 or target_shape == 0:
                continue

            mask_logits = torch.isin(idx_logits[_idx_logits], idx_targets[_idx_targets])
            mask_targets = torch.isin(idx_targets[_idx_targets], idx_logits[_idx_logits])
            pcd_logits = pcd_logits[mask_logits]
            pcd_targets = pcd_targets[mask_targets]
            if pcd_logits.shape[0] == 0 or pcd_targets.shape[0] == 0:
                continue

            perc_to_log = (pcd_logits.shape[0] / logit_shape + pcd_targets.shape[0] / target_shape) / 2
            self.log(
                "utils/points_in_common",
                perc_to_log,
                on_step=False,
                on_epoch=True,
                batch_size=len(pcd_mask_logits),
            )
            loss += self.criterion(pcd_logits, pcd_targets).mean()
            valid_pcd += 1
        return loss / max(valid_pcd, 1)

    def region_loss(self, region_logits_list, region_targets1_list, selected_region_idx, selected_region_idx1, batch_num):
        """Region swapped loss with all masks created on the logits device."""
        if len(region_logits_list) == 0:
            return torch.tensor(0.0, device=self.device)
        loss = region_logits_list[0].new_tensor(0.0)
        valid_pcd = 0
        for pcd in range(batch_num):
            pcd_logits = region_logits_list[pcd]
            pcd_targets = region_targets1_list[pcd]
            logit_shape = pcd_logits.shape[0]
            target_shape = pcd_targets.shape[0]
            if logit_shape == 0 or target_shape == 0:
                continue
            device = pcd_logits.device
            idx_logits = torch.as_tensor(selected_region_idx[pcd][1:], dtype=torch.long, device=device)
            idx_targets = torch.as_tensor(selected_region_idx1[pcd][1:], dtype=torch.long, device=device)
            idx_logits = idx_logits[:logit_shape]
            idx_targets = idx_targets[:target_shape]
            if idx_logits.numel() == 0 or idx_targets.numel() == 0:
                continue
            mask_logits = torch.isin(idx_logits, idx_targets)
            mask_targets = torch.isin(idx_targets, idx_logits)
            pcd_logits = pcd_logits[mask_logits]
            pcd_targets = pcd_targets[mask_targets]
            if pcd_logits.shape[0] == 0 or pcd_targets.shape[0] == 0:
                continue
            perc_to_log = (pcd_logits.shape[0] / logit_shape + pcd_targets.shape[0] / target_shape) / 2
            self.log("utils/regions_in_common", perc_to_log, on_step=False, on_epoch=True, batch_size=batch_num)
            loss += self.criterion(pcd_logits, pcd_targets).mean()
            valid_pcd += 1
        return loss / max(valid_pcd, 1)

    def _jaccard_index_compat(self, pred_labels, gt_labels):
        try:
            return jaccard_index(
                pred_labels,
                gt_labels,
                task="multiclass",
                num_classes=self.hparams.num_classes,
                average=None,
            )
        except TypeError:
            try:
                return jaccard_index(gt_labels, pred_labels, reduction="none")
            except TypeError:
                return jaccard_index(pred_labels, gt_labels, reduction="none")

    def validation_step(self, data, batch_idx):
        coords, feats, real_labels, _, _, _ = data
        coords = coords.int()
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()
        out = self.model(sp_tensor)
        preds = torch.cat([out["logits_lab"], out["logits_unlab"]], dim=-1)
        sorted_label_mapping_inv = dict(sorted(self.label_mapping_inv.items(), key=lambda item: item[1]))
        sorter = list(sorted_label_mapping_inv.keys())
        preds = preds[:, sorter]
        loss = self.valid_criterion(preds, real_labels.long())
        gt_labels = real_labels
        avail_labels = torch.unique(gt_labels).long()
        _, pred_labels = torch.max(torch.softmax(preds.detach(), dim=1), dim=1)
        IoU = self._jaccard_index_compat(pred_labels, gt_labels)
        IoU = IoU[avail_labels]
        self.log("valid/loss", loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
        IoU_to_log = {
            f"valid/IoU/{self.label_dict[int(avail_labels[i])]}": label_IoU
            for i, label_IoU in enumerate(IoU)
        }
        for label, value in IoU_to_log.items():
            print(label, value)
            self.log(label, value, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return loss

