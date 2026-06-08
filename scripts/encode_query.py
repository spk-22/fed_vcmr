"""
Encode a text query using MobileCLIP and send it to the phone via ADB.

Usage:
  python scripts/encode_query.py "person jumping into swimming pool"
  python scripts/encode_query.py "cooking pasta in kitchen" --no-send
"""
import argparse
import base64
import subprocess
import sys
import numpy as np
import torch
import open_clip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--no-send", action="store_true", help="Just print base64, don't send to phone")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional projection head checkpoint")
    args = parser.parse_args()

    # Load MobileCLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP-S1", pretrained="datacompdr"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")

    # Encode query
    with torch.no_grad():
        tokens = tokenizer([args.query]).to(device)
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy().astype(np.float32).flatten()

    # Optionally apply projection head
    if args.checkpoint:
        import torch.nn as nn
        import torch.nn.functional as F

        class ProjectionHead(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.linear = nn.Linear(dim, dim, bias=False)
            def forward(self, x):
                return F.normalize(self.linear(x), dim=-1)

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        head = ProjectionHead()
        head.load_state_dict(ckpt["text_head"])
        head.eval()
        with torch.no_grad():
            text_emb = head(torch.tensor(text_emb).unsqueeze(0)).numpy().flatten()

    # Encode as base64
    b64 = base64.b64encode(text_emb.tobytes()).decode("ascii")

    if args.no_send:
        print(b64)
        return

    # Send to phone
    print(f"Query: \"{args.query}\"")
    print(f"Embedding: {text_emb.shape}, norm={np.linalg.norm(text_emb):.4f}")
    print("Sending to phone...")

    # Quote the query label to prevent ADB shell word-splitting
    safe_query = args.query.replace("'", "'\\''")
    cmd = [
        "adb", "shell", "am", "broadcast",
        "-n", "com.fedvcmr/.SearchReceiver",
        "--es", "query_emb", b64,
        "--es", "query", f"'{safe_query}'"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())

    # Wait a moment then capture result
    import time
    time.sleep(2)
    result = subprocess.run(
        ["adb", "logcat", "-d", "-s", "VCMR_RESULT"],
        capture_output=True, text=True
    )
    # Print only the most recent result line
    lines = [l for l in result.stdout.strip().split("\n") if "VCMR_RESULT" in l and "query=" in l]
    if lines:
        print("\n" + lines[-1])
    else:
        # Check for errors
        result = subprocess.run(
            ["adb", "logcat", "-d", "-s", "VCMR_RESULT:E"],
            capture_output=True, text=True
        )
        print("\n" + result.stdout.strip())


if __name__ == "__main__":
    main()
