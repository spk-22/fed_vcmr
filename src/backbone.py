import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from typing import List, Union
import numpy as np

from src.config import BACKBONE_NAME, PRETRAINED_TAG

class MobileCLIPWrapper:
    def __init__(self, model_name=BACKBONE_NAME, pretrained=PRETRAINED_TAG, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        # For mobileclip_s1, the pretrained tag might be 'apple/mobileclip_s1' or similar depending on open_clip version
        # Usually 'mobileclip_s1' with 'mc_s1' works if installed from apple's repo or updated open_clip
        print(f"Loading {model_name} on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        return text_features.cpu().numpy()

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        processed_images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        image_features = self.model.encode_image(processed_images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features.cpu().numpy()

    def get_dimension(self):
        return 512
