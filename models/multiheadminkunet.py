import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import MinkowskiEngine.MinkowskiFunctional as MF
from models.minkunet import MinkUNet34C


class EP(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super(EP, self).__init__()
        self.embedding = ME.MinkowskiConvolution(
            output_dim,
            output_dim // 2,
            kernel_size=1,
            bias=False,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)
        print("Equiangular Classifier")
        P = self.generate_random_orthogonal_matrix(output_dim // 2, num_prototypes)
        I = torch.eye(num_prototypes)
        one = torch.ones(num_prototypes, num_prototypes)
        M = np.sqrt(num_prototypes / (num_prototypes - 1)) * torch.matmul(P, I - ((1 / num_prototypes) * one))


        self.M = M.cuda()
        # self.magnitude = nn.Parameter(torch.randn(1, num_prototypes))

    def generate_random_orthogonal_matrix(self, feat_in, num_prototypes):
        # feat in has to be larger than num classes.
        a = np.random.random(size=(feat_in, num_prototypes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_prototypes), atol=1e-06), torch.max(
            torch.abs(torch.matmul(P.T, P) - torch.eye(num_prototypes)))
        return P

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(x)

        x = torch.nn.functional.normalize(x.F, dim=1, p=2)
        head = torch.nn.functional.normalize(self.M, dim=0, p=2)
        logits = x @ head
        # return logits*self.magnitude
        return logits


class LinearCls(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()

        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)

    def forward(self, x):
        return self.prototypes(x).F


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()
        print("Cosine Classifier")
        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)

    def forward(self, x):
        x = torch.nn.functional.normalize(x.F, dim=1, p=2)
        head = torch.nn.functional.normalize(self.prototypes.kernel, dim=0, p=2)
        logits = x @ head
        return logits


class MultiHead(nn.Module):
    def __init__(
            self, input_dim, num_prototypes, num_heads
    ):
        super().__init__()
        self.num_heads = num_heads

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(input_dim, num_prototypes) for _ in range(num_heads)]
        )

        self.linears = torch.nn.ModuleList(
            [LinearCls(input_dim, num_prototypes) for _ in range(num_heads)]
        )

    def forward_head(self, head_idx, feats):
        return self.prototypes[head_idx](feats), self.linears[head_idx](feats), feats.F

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadMinkUnet(nn.Module):
    def __init__(
            self,
            num_labeled,
            num_unlabeled,
            overcluster_factor=None,
            num_heads=1
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34C(1, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        self.head_lab = EP(output_dim=self.feat_dim, num_prototypes=num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads
            )

        if overcluster_factor is not None:
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads
            )

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(feats)}
        if hasattr(self, "head_unlab"):
            logits_unlab, logits_unlab_linear, proj_feats_unlab = self.head_unlab(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "logits_unlab_linear": logits_unlab_linear,
                    "proj_feats_unlab": proj_feats_unlab,
                }
            )
        if hasattr(self, "head_unlab_over"):
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats.F
            return out

def geometric_aware_hypergraph_reasoning(feats, prototypes, K, alpha, device):
    """
    Implements Algorithm 1: Geometric-Aware Hypergraph Reasoning.

    Args:
        feats (ME.SparseTensor): Feature representations of the point cloud (output of the encoder).
                                 Assumes feats.F contains the feature vectors.
        prototypes (torch.Tensor): Geometry-Aware Prototypes {P1, P2, ..., PN}.  Shape: (N, feature_dim).
                                  This should be a tensor on the same device as feats.F.
        K (int): Hyperparameter for selecting nearest neighbors.
        alpha (float): Similarity weighting coefficient.
        device (torch.device): The device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: (updated_hyperedges, pseudo_labels)
            - updated_hyperedges (list of lists): Updated hypergraph structure.  Each inner list represents a hyperedge.
            - pseudo_labels (torch.Tensor): Pseudo-labels for novel classes.
    """

    # Ensure prototypes are on the correct device
    prototypes = prototypes.to(device)

    # 1. Geometry-Semantic Dual Similarity Calculation:
    Sb = compute_geometric_semantic_similarity(feats, prototypes, alpha, device)  # (N, N) similarity matrix

    # 3. Multi-Class Collaborative Hyperedge Construction:
    hyperedges = construct_hyperedges(Sb, K)

    # 5. Dynamic Hyperedge Update:
    updated_hyperedges = dynamic_hyperedge_update(feats, prototypes, Sb, K, device)

    # 7. Hypergraph for Novel Class Discovery:
    pseudo_labels = hypergraph_convolution(feats, updated_hyperedges, prototypes, device)

    return updated_hyperedges, pseudo_labels


def compute_geometric_semantic_similarity(feats, prototypes, alpha, device):
    """
    Computes the geometric-semantic similarity between prototypes.

    Args:
        feats (ME.SparseTensor): Feature representations of the point cloud.
        prototypes (torch.Tensor): Geometry-Aware Prototypes.
        alpha (float): Similarity weighting coefficient.
        device (torch.device): The device to run computations on.

    Returns:
        torch.Tensor: Similarity matrix Sb (N, N).
    """
    # Ensure feats.F and prototypes are on the same device
    feats_F = feats.F.to(device)
    prototypes = prototypes.to(device)

    # Semantic Similarity (Ss): Cosine similarity between prototypes
    prototypes_normalized = torch.nn.functional.normalize(prototypes, dim=1)
    Ss = torch.matmul(prototypes_normalized, prototypes_normalized.transpose(0, 1))

    # Geometric Similarity (Sg):  Average cosine similarity between point features and prototypes
    # For each prototype, compute the average cosine similarity between the prototype and all point features in feats.
    Sg = torch.zeros(prototypes.shape[0], prototypes.shape[0], device=device)
    feats_normalized = torch.nn.functional.normalize(feats_F, dim=1)

    for i in range(prototypes.shape[0]):
        prototype_i_normalized = prototypes_normalized[i, :]
        Sg[i, :] = torch.mean(torch.matmul(feats_normalized, prototype_i_normalized.unsqueeze(0).transpose(0, 1)), dim=0)

    beta = 1 - alpha
    Sb = alpha * Sg + beta * Ss
    return Sb


def construct_hyperedges(Sb, K):
    """
    Constructs hyperedges based on the similarity matrix Sb.

    Args:
        Sb (torch.Tensor): Similarity matrix (N, N).
        K (int): Number of nearest neighbors to include in each hyperedge.

    Returns:
        list of lists: Hyperedges. Each inner list contains the indices of the prototypes in the hyperedge.
    """
    hyperedges = []
    for i in range(Sb.shape[0]):
        # Select K nearest prototypes based on Sb(Pi, .)
        _, topk_indices = torch.topk(Sb[i, :], K)  # Get indices of K nearest neighbors
        hyperedge = [i] + topk_indices.tolist()  # Hyperedge: {Pi, Pj1, Pj2, ..., Pjk}
        hyperedges.append(hyperedge)
    return hyperedges


def dynamic_hyperedge_update(feats, prototypes, Sb, K, device):
    """
    Dynamically updates hyperedges based on the current batch.

    Args:
        feats (ME.SparseTensor): Feature representations of the point cloud.
        prototypes (torch.Tensor): Geometry-Aware Prototypes.
        Sb (torch.Tensor): Similarity matrix (N, N).
        K (int): Number of nearest neighbors to include in each hyperedge.
        device (torch.device): The device to run computations on.

    Returns:
        list of lists: Updated hyperedges.
    """
    updated_hyperedges = []
    for i in range(Sb.shape[0]):
        # Update similarity Sb(Pi, .) based on the current batch (using the same similarity calculation)
        # In a real implementation, you might want to update Sb incrementally instead of recomputing it entirely.
        # Select top-k related prototypes using the updated similarity
        _, topk_indices = torch.topk(Sb[i, :], K)
        updated_hyperedge = [i] + topk_indices.tolist()
        updated_hyperedges.append(updated_hyperedge)
    return updated_hyperedges


def hypergraph_convolution(feats, hyperedges, prototypes, device, num_layers=2):
    """
    Performs hypergraph convolution to compute pseudo-labels for novel classes.

    Args:
        feats (ME.SparseTensor): Feature representations of the point cloud.
        hyperedges (list of lists): Updated hyperedges.
        prototypes (torch.Tensor): Geometry-Aware Prototypes.
        device (torch.device): The device to run computations on.
        num_layers (int): Number of hypergraph convolution layers.

    Returns:
        torch.Tensor: Pseudo-labels for novel classes.
    """

    # Initialize H(0) with prototype features
    H = prototypes.clone().detach().to(device)  # (N, feature_dim)

    # Hypergraph Convolution Layers
    for l in range(num_layers):
        # Construct the incidence matrix A
        A = construct_incidence_matrix(hyperedges, H.shape[0]).to(device)  # (N, num_hyperedges)

        # Compute the degree matrices D_v and D_e
        D_v = torch.diag(torch.sum(A, dim=1))  # Vertex degrees (N, N)
        D_e = torch.diag(torch.sum(A, dim=0))  # Hyperedge degrees (num_hyperedges, num_hyperedges)

        # Compute the inverse degree matrices (add a small epsilon for numerical stability)
        D_v_inv = torch.inverse(D_v + torch.eye(D_v.shape[0], device=device) * 1e-8)
        D_e_inv = torch.inverse(D_e + torch.eye(D_e.shape[0], device=device) * 1e-8)

        # Hypergraph Convolution operation
        W = torch.nn.Linear(H.shape[1], H.shape[1]).to(device) # Weight matrix for the layer
        H = torch.relu(torch.matmul(D_v_inv, torch.matmul(A, torch.matmul(D_e_inv, torch.matmul(A.transpose(0, 1), H))))) # Hypergraph convolution
        H = W(H)

    # Compute pseudo-labels as: ˆyuk = softmax(H(L)uk · W(L))
    W_L = torch.nn.Linear(H.shape[1], prototypes.shape[0]).to(device) # Weight matrix for the last layer
    pseudo_labels = torch.softmax(W_L(H), dim=1)

    return pseudo_labels


def construct_incidence_matrix(hyperedges, num_vertices):
    """
    Constructs the incidence matrix for the hypergraph.

    Args:
        hyperedges (list of lists): Hyperedges.
        num_vertices (int): Number of vertices in the hypergraph.

    Returns:
        torch.Tensor: Incidence matrix A (num_vertices, num_hyperedges).
    """
    num_hyperedges = len(hyperedges)
    A = torch.zeros(num_vertices, num_hyperedges)

    for j, hyperedge in enumerate(hyperedges):
        for vertex_index in hyperedge:
            A[vertex_index, j] = 1

    return A


class MinkUnet(nn.Module):
    def __init__(
            self,
            num_labeled,
            num_unlabeled,
            discover=True,
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34C(1, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        self.head_lab = Prototypes(output_dim=self.feat_dim,
                                   num_prototypes=num_labeled)
        if discover:
            self.head_unlab = Prototypes(output_dim=self.feat_dim,
                                         num_prototypes=num_unlabeled)

            self.head_unlab_linears = EP(output_dim=self.feat_dim, num_prototypes=num_unlabeled)

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(feats)}
        if hasattr(self, "head_unlab"):
            logits_unlab = self.head_unlab(feats)
            logits_unlab_linear = self.head_unlab_linears(feats)
            proj_feats_unlab = feats.F
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "logits_unlab_linear": logits_unlab_linear,
                    "proj_feats_unlab": proj_feats_unlab,
                }
            )

        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            #import ipdb;ipdb.set_trace()
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            #import ipdb;
            #ipdb.set_trace()

            out = self.forward_heads(feats)
            out["feats"] = feats.F
            return out