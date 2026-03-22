import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from src.backbone import MobileCLIPWrapper
from src.config import DEVICE
from src.model import apply_projection
from src.adapter import apply_adapter

class QueryService:
    def __init__(self, backbone: MobileCLIPWrapper = None):
        self.backbone = backbone or MobileCLIPWrapper(device=DEVICE)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string or multiple phrasings."""
        # Use backbone's encode_text which already normalizes and returns numpy
        return self.encode_queries([text])

    def expand_query(self, text: str) -> List[str]:
        """Simple rule-based query expander (placeholders for now)."""
        # In a real scenario, this would use a LLM or synonym list
        # For now, we manually mock the expansion logic as requested
        phrasings = [text]
        # Example expansion (very basic/manual for this milestone)
        if "riding bicycle" in text.lower():
            phrasings.append(text.lower().replace("riding bicycle", "cycling"))
            phrasings.append(text.lower().replace("riding bicycle", "individual on a bike"))
        elif "cat" in text.lower():
            phrasings.append(text.lower().replace("cat", "feline"))
            phrasings.append("a small animal purring")
        else:
            # Generic placeholders
            phrasings.append(f"a video of {text}")
            phrasings.append(f"showing {text}")
            
        return phrasings[:3]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        # Get backbone embeddings (numpy)
        embs_np = self.backbone.encode_text(texts).astype(np.float32)

        # Apply projection if verified
        with torch.no_grad():
            tensor = torch.tensor(embs_np, device=DEVICE)
            projected = apply_projection(tensor)

            # Apply adapter (after projection)
            adapted = apply_adapter(projected, client_id="client_1")

            return adapted.cpu().numpy()
