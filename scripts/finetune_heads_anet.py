# scripts/finetune_heads_anet.py
# Fine-tune projection heads on ActivityNet captions
# Uses anet_data_consolidated.pt which you already have

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS    = 20
LR        = 1e-5   # small — identity init, preserve MSR-VTT knowledge
BATCH     = 64

class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

def infonce(v, t, temp=0.07):
    logits   = torch.matmul(t, v.T) / temp
    labels   = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2

class AnetPairsDataset(Dataset):
    def __init__(self, data_path):
        raw       = torch.load(data_path, weights_only=False)
        self.items = [(r['features'], r['query_embed']) for r in raw]
        print(f'ANet pairs: {len(self.items):,}')
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        frames, text = self.items[i]
        w    = torch.tensor([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5])
        w    = w / w.sum()
        v    = (frames * w.unsqueeze(-1)).sum(0)
        v    = F.normalize(v, dim=-1)
        t    = F.normalize(text, dim=-1)
        return v, t

# Load existing heads
ckpt        = torch.load('checkpoints/best_model.pt', map_location=DEVICE)
vision_head = ProjectionHead().to(DEVICE)
text_head   = ProjectionHead().to(DEVICE)
vision_head.load_state_dict(ckpt['vision_head'])
text_head.load_state_dict(ckpt['text_head'])
print('MSR-VTT heads loaded.')

dataset = AnetPairsDataset('cache/anet_data_consolidated.pt')
loader  = DataLoader(dataset, batch_size=BATCH,
                     shuffle=True, num_workers=0)

optimizer = optim.AdamW(
    list(vision_head.parameters()) +
    list(text_head.parameters()),
    lr=LR, weight_decay=0.0
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-7
)

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    vision_head.train()
    text_head.train()
    total = 0.0

    for v, t in loader:
        v = v.to(DEVICE)
        t = t.to(DEVICE)
        loss = infonce(vision_head(v), text_head(t))
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vision_head.parameters()) +
            list(text_head.parameters()), 0.5
        )
        optimizer.step()
        total += loss.item()

    scheduler.step()
    avg = total / len(loader)
    print(f'Epoch {epoch:2d}/{EPOCHS} | Loss: {avg:.4f}')

    if avg < best_loss:
        best_loss = avg
        torch.save({
            'vision_head': vision_head.state_dict(),
            'text_head':   text_head.state_dict(),
            'arch':        'identity_linear_512_anet_finetuned'
        }, 'checkpoints/best_model_anet.pt')
        print(f'  Saved best -> checkpoints/best_model_anet.pt')

print(f'\nDone. Run eval_anet_vcmr.py with --proj_heads checkpoints/best_model_anet.pt')
