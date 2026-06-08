# scripts/export_pt_mobile.py
"""
M33: PyTorch Mobile Export
Exports FedVCMR models directly to PyTorch Lite (.ptl) format for Android.
This bypasses TFLite/ONNX compatibility issues on Windows by utilizing
the native org.pytorch:pytorch_android_lite Android dependency.

Components Exported:
1. VCMR Text Head (512 -> 512, L2 norm)
2. VCMR Vision Head (512 -> 512, L2 norm)
3. FL Text Head (512 -> 512, L2 norm)
4. FL Vision Head (512 -> 512, L2 norm)
5. Temporal Grounding Transformer (Cross-Modal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.mobile_optimizer import optimize_for_mobile

DEVICE = 'cpu'
OUT_DIR = 'checkpoints/export'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Model Definitions ──────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

class CrossModalTransformer(nn.Module):
    def __init__(self, visual_dim=512, query_dim=512, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.query_proj  = nn.Linear(query_dim, hidden_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        encoder_layer    = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                           dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                           nn.Linear(hidden_dim, 2), nn.Sigmoid())
    def forward(self, v, q):
        v = self.visual_proj(v) + self.pos_embed
        q = self.query_proj(q).unsqueeze(1)
        tokens = torch.cat([q, v], dim=1)
        transformed = self.transformer(tokens)
        return self.regressor(transformed[:, 0, :])

# ── 2. Export Helper ──────────────────────────────────────────────
def export_to_ptl(model, dummy_args, name):
    ptl_path = os.path.join(OUT_DIR, f"{name}.ptl")
    print(f"\nExporting {name} to PyTorch Lite...")
    try:
        # 1. Trace the model
        traced_script_module = torch.jit.trace(model, dummy_args)
        # 2. Optimize for mobile
        optimized_traced_model = optimize_for_mobile(traced_script_module)
        # 3. Save as .ptl
        optimized_traced_model._save_for_lite_interpreter(ptl_path)
        print(f"  -> Successfully saved to {ptl_path}")
        print(f"  -> Original size: {os.path.getsize(ptl_path)/1024:.1f} KB")
    except Exception as e:
        print(f"  -> ERROR converting {name}: {e}")

# ── 3. Main Export Routine ────────────────────────────────────────
def main():
    print(f"{'='*50}\nM33: EXPORTING MODELS TO PYTORCH LITE (.ptl)\n{'='*50}")

    dummy_1d = (torch.randn(1, 512),)

    # A. Projection Heads (VCMR)
    ckpt = torch.load('checkpoints/best_model_anet_fixed.pt', map_location='cpu', weights_only=False)
    
    text_head = ProjectionHead().eval()
    text_head.load_state_dict(ckpt['text_head'])
    export_to_ptl(text_head, dummy_1d, 'vcmr_text_head')

    vision_head = ProjectionHead().eval()
    vision_head.load_state_dict(ckpt['vision_head'])
    export_to_ptl(vision_head, dummy_1d, 'vcmr_vision_head')

    # B. Projection Heads (FL Global)
    fl_ckpt = torch.load('checkpoints/fl_global_model.pt', map_location='cpu', weights_only=False)
    
    fl_text_head = ProjectionHead().eval()
    fl_text_head.load_state_dict(fl_ckpt['text_head'])
    export_to_ptl(fl_text_head, dummy_1d, 'fl_text_head')

    fl_vision_head = ProjectionHead().eval()
    fl_vision_head.load_state_dict(fl_ckpt['vision_head'])
    export_to_ptl(fl_vision_head, dummy_1d, 'fl_vision_head')

    # C. Temporal Transformer
    tg_ckpt = torch.load('checkpoints/temporal_grounding_best.pt', map_location='cpu', weights_only=False)
    transformer = CrossModalTransformer().eval()
    state_dict = tg_ckpt['model'] if isinstance(tg_ckpt, dict) and 'model' in tg_ckpt else tg_ckpt
    transformer.load_state_dict(state_dict)
    
    dummy_v = torch.randn(1, 8, 512)
    dummy_q = torch.randn(1, 512)
    export_to_ptl(transformer, (dummy_v, dummy_q), 'temporal_transformer')

    print("\nDONE.")

if __name__ == '__main__':
    main()
