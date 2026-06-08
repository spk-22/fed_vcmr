import torch
import torch.nn as nn
import open_clip
import os

# Constants
MODEL_NAME = "MobileCLIP-S1"
PRETRAINED = "datacompdr"
ASSET_DIR = "android/app/src/main/assets/models"

print(f"Loading {MODEL_NAME}...")
model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model.eval()

# 1. Export Text Backbone (PTL)
print("Exporting Text PTL...")
class TextBackbone(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
    def forward(self, tokens):
        return self.clip_model.encode_text(tokens, normalize=True)

text_pt = TextBackbone(model).eval()
tokens = open_clip.get_tokenizer(MODEL_NAME)(["person"])
traced_text = torch.jit.trace(text_pt, tokens, strict=False, check_trace=False)
traced_text._save_for_lite_interpreter(os.path.join(ASSET_DIR, "mobileclip_text_encoder.ptl"))

# 2. Export Vision Backbone (PTL)
print("Exporting Vision PTL...")
class VisionBackbone(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
    def forward(self, x):
        # Return raw features, Java will normalize
        return self.visual(x)

vision_pt = VisionBackbone(model).eval()
img = torch.randn(1, 3, 256, 256)
traced_vision = torch.jit.trace(vision_pt, img, strict=False, check_trace=False)
traced_vision._save_for_lite_interpreter(os.path.join(ASSET_DIR, "mobileclip_vision_encoder.ptl"))

print("Done. Both backbones exported as .ptl")
