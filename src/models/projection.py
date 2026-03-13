import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    2-layer MLP with residual connection.
    MLP output layer is zero-initialized so initial behavior = pure residual,
    preserving CLIP alignment at the start of training.
    """
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
        # Residual downsample — preserves CLIP alignment
        self.residual = nn.Linear(in_dim, out_dim, bias=False)

        # Zero-init MLP output so initial output ≈ residual(x)
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # output = learned refinement + preserved CLIP structure
        return F.normalize(self.net(x) + self.residual(x), dim=-1)
