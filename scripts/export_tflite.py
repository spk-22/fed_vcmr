# scripts/export_tflite.py
"""
FEDVCMR: TFLite Model Export via ONNX Intermediate
Exports 8 models for Android deployment:
1. dgse.tflite
2. temporal_grounding.tflite
3. text_head_fl_global_model.tflite
4. vision_head_fl_global_model.tflite
5. text_head_best_model.tflite
6. vision_head_best_model.tflite
7. text_head_best_model_anet_fixed.tflite
8. vision_head_best_model_anet_fixed.tflite
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import subprocess
import glob
import shutil

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dgse import DGSE

DEVICE = 'cpu'
OUT_DIR = 'checkpoints/export'
os.makedirs(OUT_DIR, exist_ok=True)

# Set environment for onnx2tf to use legacy Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
def export_to_tflite(model, dummy_args, name):
    onnx_path = os.path.join(OUT_DIR, f"{name}.onnx")
    tflite_dir = os.path.join(OUT_DIR, f"{name}_tflite")
    tflite_path = os.path.join(OUT_DIR, f"{name}.tflite")
    
    # Cleanup old artifacts
    if os.path.exists(tflite_dir): shutil.rmtree(tflite_dir)
    
    print(f"\nExporting {name} to ONNX...")
    try:
        if isinstance(dummy_args, tuple):
            input_names = [f"input_{i}" for i in range(len(dummy_args))]
        else:
            input_names = ["input"]
            dummy_args = (dummy_args,)

        torch.onnx.export(
            model, dummy_args, onnx_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=14,
            do_constant_folding=True
        )
        print(f"  -> Successfully saved to {onnx_path}")
        
        print(f"Converting {name}.onnx to TFLite via onnx2tf...")
        # Use subprocess to run onnx2tf with legacy Keras
        cmd = ["onnx2tf", "-i", onnx_path, "-o", tflite_dir, "--non_verbose"]
        subprocess.run(cmd, check=False)
        
        # onnx2tf creates files like {name}_float32.tflite
        tflite_files = glob.glob(os.path.join(tflite_dir, "*_float32.tflite"))
        if not tflite_files:
            tflite_files = glob.glob(os.path.join(tflite_dir, "*.tflite"))
            
        if tflite_files:
            shutil.copy2(tflite_files[0], tflite_path)
            print(f"  -> Successfully saved to {tflite_path}")
        else:
            print(f"  -> ERROR: Could not find converted TFLite in {tflite_dir}")

    except Exception as e:
        print(f"  -> ERROR exporting {name}: {e}")

# ── 3. Main Export Routine ────────────────────────────────────────
def main():
    print(f"{'='*50}\nFEDVCMR: EXPORTING 8 MODELS TO TFLITE (ONNX PATH)\n{'='*50}")

    dummy_1d = (torch.randn(1, 512),)
    dummy_v = torch.randn(1, 8, 512)
    dummy_q = torch.randn(1, 512)

    # 1. DGSE
    print("\n[1/8] dgse.tflite")
    dgse_ckpt = torch.load('checkpoints/dgse_best.pt', map_location='cpu', weights_only=False)
    dgse = DGSE().eval()
    dgse.load_state_dict(dgse_ckpt['dgse'] if 'dgse' in dgse_ckpt else dgse_ckpt)
    export_to_tflite(dgse, (dummy_v, dummy_q), 'dgse')

    # 2. Temporal Grounding
    print("\n[2/8] temporal_grounding.tflite")
    tg_ckpt = torch.load('checkpoints/temporal_grounding_best.pt', map_location='cpu', weights_only=False)
    transformer = CrossModalTransformer().eval()
    tg_state = tg_ckpt['model'] if isinstance(tg_ckpt, dict) and 'model' in tg_ckpt else tg_ckpt
    transformer.load_state_dict(tg_state)
    export_to_tflite(transformer, (dummy_v, dummy_q), 'temporal_grounding')

    # 3. FL Global (Text)
    print("\n[3/8] text_head_fl_global_model.tflite")
    fl_ckpt = torch.load('checkpoints/fl_global_model.pt', map_location='cpu', weights_only=False)
    fl_text_head = ProjectionHead().eval()
    fl_text_head.load_state_dict(fl_ckpt['text_head'])
    export_to_tflite(fl_text_head, dummy_1d, 'text_head_fl_global_model')

    # 4. FL Global (Vision)
    print("\n[4/8] vision_head_fl_global_model.tflite")
    fl_vision_head = ProjectionHead().eval()
    fl_vision_head.load_state_dict(fl_ckpt['vision_head'])
    export_to_tflite(fl_vision_head, dummy_1d, 'vision_head_fl_global_model')

    # 5. Best Model (Text)
    print("\n[5/8] text_head_best_model.tflite")
    bm_ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    bm_text_head = ProjectionHead().eval()
    bm_text_head.load_state_dict(bm_ckpt['text_head'])
    export_to_tflite(bm_text_head, dummy_1d, 'text_head_best_model')

    # 6. Best Model (Vision)
    print("\n[6/8] vision_head_best_model.tflite")
    bm_vision_head = ProjectionHead().eval()
    bm_vision_head.load_state_dict(bm_ckpt['vision_head'])
    export_to_tflite(bm_vision_head, dummy_1d, 'vision_head_best_model')

    # 7. ANet Fixed (Text)
    print("\n[7/8] text_head_best_model_anet_fixed.tflite")
    anet_ckpt = torch.load('checkpoints/best_model_anet_fixed.pt', map_location='cpu', weights_only=False)
    anet_text_head = ProjectionHead().eval()
    anet_text_head.load_state_dict(anet_ckpt['text_head'])
    export_to_tflite(anet_text_head, dummy_1d, 'text_head_best_model_anet_fixed')

    # 8. ANet Fixed (Vision)
    print("\n[8/8] vision_head_best_model_anet_fixed.tflite")
    anet_vision_head = ProjectionHead().eval()
    anet_vision_head.load_state_dict(anet_ckpt['vision_head'])
    export_to_tflite(anet_vision_head, dummy_1d, 'vision_head_best_model_anet_fixed')

    print("\nDONE.")

if __name__ == '__main__':
    main()
