# scripts/debug_embeddings.py
import torch, open_clip, numpy as np
import torch.nn as nn, torch.nn.functional as F

DEVICE = 'cpu'

model, _, _ = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr'
)
model.eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

# Check zero-shot alignment (no projection heads)
with torch.no_grad():
    t = tokenizer(["a man is driving a car"]).to(DEVICE)
    t_emb = model.encode_text(t)
    t_emb = F.normalize(t_emb, dim=-1)

# Load one chunk from cache
import sqlite3
cache = np.memmap('cache/frame_features.bin',
                  dtype='float16', mode='r',
                  shape=(32216, 8, 512))
conn  = sqlite3.connect('fedvcmr.db')
# Get a driving video chunk
row   = conn.execute(
    "SELECT rowid-1 FROM chunks WHERE video_id='video0' LIMIT 1"
).fetchone()[0]
conn.close()

frames = torch.tensor(cache[row].astype('float32'))
w      = torch.tensor([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5])
w      = w / w.sum()
v_emb  = F.normalize((frames * w[:,None]).sum(0).unsqueeze(0), dim=-1)

sim_zero_shot = (t_emb @ v_emb.T).item()
print(f'Zero-shot cosine sim (driving query, video0): {sim_zero_shot:.4f}')
# Expect: > 0.20 (positive alignment)

# Now check with broken projection head
ckpt = torch.load('checkpoints/proj_heads_final.pt', map_location='cpu')

class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512,512), nn.LayerNorm(512),
            nn.GELU(), nn.Linear(512,256)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

vh = ProjectionHead(); vh.load_state_dict(ckpt['vision_head']); vh.eval()
th = ProjectionHead(); th.load_state_dict(ckpt['text_head']);   th.eval()

with torch.no_grad():
    v_proj = vh(v_emb)
    t_proj = th(t_emb)
    sim_trained = (t_proj @ v_proj.T).item()

print(f'Trained cosine sim  (driving query, video0): {sim_trained:.4f}')
# If negative or near zero → confirms projection head destroyed alignment
