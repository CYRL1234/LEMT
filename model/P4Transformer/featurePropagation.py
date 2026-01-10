import torch
import torch.nn as nn

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils
from typing import List

class FeaturePropagation(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k  # Follow paper's choice of k=3 nearest neighbors

    def forward(self, xyz_subsampled, feats_subsampled, xyz_original):
        """
        Args (per paper's 4D semantic segmentation branch):
            xyz_subsampled: [B, L, N_sub, 3] → Subsampled coordinates (after P4Conv)
            feats_subsampled: [B, L, N_sub, dim] → Transformer outputs (low-res features)
            xyz_original: [B, L, N, 3] → Original high-res coordinates (input to P4Conv)
        Returns:
            feats_propagated: [B, L, N, dim] → High-res features for key point prediction
        """
        B, L, N, _ = xyz_original.shape  # B=batch, L=frames, N=original points per frame
        _, _, N_sub, dim = feats_subsampled.shape  # N_sub=subsampled points per frame
        feats_propagated = []

        # Process each batch and frame individually (matches paper's per-frame processing)
        for b in range(B):
            for t in range(L):
                # Extract single batch-frame tensors (reduce dimensions for clarity)
                xyz_o = xyz_original[b, t]  # [N, 3] → Original points for this frame
                xyz_s = xyz_subsampled[b, t]  # [N_sub, 3] → Subsampled points for this frame
                feats_s = feats_subsampled[b, t]  # [N_sub, dim] → Low-res features for this frame

                # Step 1: Compute Euclidean distance between original and subsampled points (paper Eq. 5)
                # Shape: [N, N_sub] → Distance from each original point to each subsampled point
                dist = torch.cdist(xyz_o, xyz_s)

                # Step 2: Get indices of k-nearest subsampled points (paper uses k=3)
                # Shape: [N, k] → For each original point, indices of k closest subsampled points
                _, idx = dist.topk(self.k, dim=-1, largest=False)  # "largest=False" → smallest distances

                # Step 3: Gather features of k-nearest subsampled points (fix dimension mismatch here)
                # Input to gather: [N_sub, dim] (2D)
                # Index tensor: [N, k] (2D) → Need to expand to [N, k, 1] (3D) to match gather's input dims
                # After gather: [N, k, dim] → Features of k neighbors for each original point
                idx_expanded = idx.unsqueeze(-1).expand(-1, -1, dim)  # [N, k, dim] (3D, same as input+1 dim)
                feats_neighbors = torch.gather(
                    feats_s.unsqueeze(0).expand(N, -1, -1),  # Input: [N, N_sub, dim] (3D, broadcast over N)
                    dim=1,  # Gather along the "subsampled point" dimension (N_sub)
                    index=idx_expanded  # Index: [N, k, dim] (3D, same dims as input)
                )

                # Step 4: Compute inverse distance weights (paper Eq. 5: w=1/||δ||²)
                # Shape: [N, k] → Weights for each of the k neighbors
                dist_neighbors = torch.gather(dist, dim=1, index=idx)  # [N, k] → Distances of k neighbors
                weights = 1.0 / (dist_neighbors ** 2 + 1e-6)  # Avoid division by zero (paper's stability trick)
                weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize weights to sum to 1

                # Step 5: Weighted sum of neighbor features (paper Eq. 5)
                # Shape: [N, dim] → Recovered high-res feature for each original point
                feats_p = (feats_neighbors * weights.unsqueeze(-1)).sum(dim=1)

                # Collect results for this batch-frame
                feats_propagated.append(feats_p)

        # Reshape to match input batch-frame structure: [B, L, N, dim]
        feats_propagated = torch.stack(feats_propagated, dim=1).reshape(B, L, N, dim)
        return feats_propagated

