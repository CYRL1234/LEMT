import torch
import torch.nn as nn

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class KeyPointRegressionHead(nn.Module):
    def __init__(self, dim, num_keypoints=17):
        super().__init__()
        self.num_keypoints = num_keypoints  # e.g., 17 human body joints
        # Output: (confidence + 3 coordinates) per key point → 4 × num_keypoints dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4 * self.num_keypoints)  # 4: confidence (0-1) + x,y,z
        )
    
    def forward(self, feats):
        """
        Args:
            feats: [B, L, N, dim] — propagated high-res features
        Returns:
            keypoints: [B, L, num_keypoints, 4] — (confidence, x, y, z) per key point
        """
        B, L, N, dim = feats.shape
        output = self.mlp(feats)  # [B, L, N, 4×num_keypoints]
        output = output.reshape(B, L, N, self.num_keypoints, 4)  # [B, L, N, K, 4]
        
        # For sparse key points: select the point with highest confidence per key point
        confidences = output[..., 0]  # [B, L, N, K]
        max_conf_idx = confidences.argmax(dim=2, keepdim=True)  # [B, L, 1, K]
        
        # Gather coordinates for the most confident point per key point
        keypoints = torch.gather(output, 2, max_conf_idx.unsqueeze(-1).expand(B, L, 1, self.num_keypoints, 4))  # [B, L, 1, K, 4]
        keypoints = keypoints.squeeze(2)  # [B, L, K, 4] (confidence, x, y, z)
        return keypoints

