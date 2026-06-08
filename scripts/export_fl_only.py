
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
    print(f"{'='*50}\nFEDVCMR: EXPORTING FL GLOBAL MODELS TO TFLITE\n{'='*50}")

    dummy_1d = (torch.randn(1, 512),)

    # 3. FL Global (Text)
    print("\n[1/2] text_head_fl_global_model.tflite")
    fl_ckpt = torch.load('checkpoints/fl_global_model.pt', map_location='cpu', weights_only=False)
    fl_text_head = ProjectionHead().eval()
    fl_text_head.load_state_dict(fl_ckpt['text_head'])
    export_to_tflite(fl_text_head, dummy_1d, 'text_head_fl_global_model')

    # 4. FL Global (Vision)
    print("\n[2/2] vision_head_fl_global_model.tflite")
    fl_vision_head = ProjectionHead().eval()
    fl_vision_head.load_state_dict(fl_ckpt['vision_head'])
    export_to_tflite(fl_vision_head, dummy_1d, 'vision_head_fl_global_model')

    print("\nDONE.")

if __name__ == '__main__':
    main()
