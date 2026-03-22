import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    Trainable projection head to be inserted after the frozen backbone.
    Architecture: Linear(512->512) -> ReLU -> Linear(512->512) -> L2 Norm
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super(ProjectionHead, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass with L2 normalization on output.
        Args:
            x (torch.Tensor): Input embeddings (batch_size, input_dim)
        Returns:
            torch.Tensor: Projected and normalized embeddings (batch_size, output_dim)
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return F.normalize(x, p=2, dim=-1)

# Helper function for safe integration
_PROJECTION_HEAD = None

def get_projection_head():
    """Singleton loader for ProjectionHead."""
    global _PROJECTION_HEAD
    if _PROJECTION_HEAD is None:
        from src.config import DEVICE, CACHE_ROOT

        # Always load model structure
        print(f"Loading ProjectionHead on {DEVICE}...")
        _PROJECTION_HEAD = ProjectionHead().to(DEVICE)

        # Load weights if available
        weights_path = CACHE_ROOT / "projection_head.pth"
        if weights_path.exists():
            print(f"Found trained weights at {weights_path}. Loading...")
            state_dict = torch.load(weights_path, map_location=DEVICE)
            _PROJECTION_HEAD.load_state_dict(state_dict)
        else:
            print("No trained weights found. Initializing randomly.")

        _PROJECTION_HEAD.eval()
    return _PROJECTION_HEAD

def apply_projection(tensor):
    """Apply projection if enabled, otherwise return tensor."""
    from src.config import USE_PROJECTION
    if USE_PROJECTION:
        head = get_projection_head()
        return head(tensor)
    return tensor
