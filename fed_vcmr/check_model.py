import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model import ProjectionHead

def test_model():
    print("Testing ProjectionHead...")
    model = ProjectionHead()
    print("Model instantiated.")

    x = torch.randn(5, 512)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check normalization
    norms = torch.norm(y, p=2, dim=-1)
    print(f"Output norms: {norms}")

    is_normalized = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"Normalized accurately: {is_normalized}")

    if is_normalized:
        print("PASS: Output is L2 normalized.")
    else:
        print("FAIL: Output is NOT normalized.")

if __name__ == "__main__":
    test_model()

