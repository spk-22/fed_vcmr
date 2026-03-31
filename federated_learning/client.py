# federated_learning/client.py
"""
Flower (flwr) Client for Federated Learning.
Handles local on-device training for a specific data shard.
"""
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import numpy as np
import os
import open_clip
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model Definition (Heads Only) ───────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

# ── InfoNCE Loss ────────────────────────────────────────────────
def info_nce_loss(v, t, temp=0.07):
    # v, t: (Batch, 512)
    sims = (v @ t.T) / temp
    labels = torch.arange(v.size(0)).to(v.device)
    loss_v = F.cross_entropy(sims, labels)
    loss_t = F.cross_entropy(sims.T, labels)
    return (loss_v + loss_t) / 2

# ── Flower Client ───────────────────────────────────────────────
class VCMRClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.text_head = ProjectionHead().to(DEVICE)
        self.vision_head = ProjectionHead().to(DEVICE)
        
        # Load Shard Data
        conn = sqlite3.connect('fedvcmr.db')
        self.vids = [r[0] for r in conn.execute(
            'SELECT video_id FROM fl_shards WHERE client_id = ?', (client_id,)
        ).fetchall()]
        
        # Pre-load features (for simulation speed)
        conn = sqlite3.connect('fedvcmr.db')
        n_total = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        self.cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(n_total, 8, 512))
        
        # Load categories for DAPHW analysis later
        self.cats = dict(conn.execute('SELECT video_id, (SELECT caption FROM captions WHERE video_id = v.video_id GROUP BY video_id LIMIT 1) FROM videos v').fetchall())
        conn.close()
        
        # Pre-load CLIP for on-device embedding (Simulation only)
        # In a real app, this would be a TFLite blob
        self.backbone, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
        self.backbone = self.backbone.to(DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
        
        # Feature Mapping (Vision)
        conn = sqlite3.connect('fedvcmr.db')
        all_chunks = conn.execute('SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
        self.idx_map = {cid: i for i, (cid, _) in enumerate(all_chunks)}
        self.vid_chunks = {}
        for cid, vid in all_chunks:
            self.vid_chunks.setdefault(vid, []).append(cid)
        conn.close()

        print(f"Client {client_id} initialized with {len(self.vids)} videos.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.text_head.state_dict().values()] + \
               [val.cpu().numpy() for val in self.vision_head.state_dict().values()]

    def set_parameters(self, parameters):
        # Split params back to text and vision heads
        mid = len(parameters) // 2
        text_params = parameters[:mid]
        vision_params = parameters[mid:]
        
        t_keys = self.text_head.state_dict().keys()
        self.text_head.load_state_dict({k: torch.tensor(v) for k, v in zip(t_keys, text_params)})
        
        v_keys = self.vision_head.state_dict().keys()
        self.vision_head.load_state_dict({k: torch.tensor(v) for k, v in zip(v_keys, vision_params)})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Keep global copy for proximal regularization
        import copy
        t_head_global = copy.deepcopy(self.text_head)
        v_head_global = copy.deepcopy(self.vision_head)
        
        optimizer = torch.optim.AdamW(list(self.text_head.parameters()) + list(self.vision_head.parameters()), 
                                    lr=1e-5, weight_decay=0.05)
        
        self.text_head.train()
        self.vision_head.train()
        
        BATCH_SIZE = 64
        mu = 0.1 # Proximal regularization strength
        total_loss = 0
        pbar = tqdm(range(0, len(self.vids), BATCH_SIZE), desc=f"Client {self.client_id} Round")
        
        for i in pbar:
            batch_vids = self.vids[i:i+BATCH_SIZE]
            batch_caps = [self.cats.get(v, "a video") for v in batch_vids]
            
            # 1. Text Embedding
            tokens = self.tokenizer(batch_caps).to(DEVICE)
            with torch.no_grad():
                t_raw = F.normalize(self.backbone.encode_text(tokens).float(), dim=-1)
            t_emb = self.text_head(t_raw)
            
            # 2. Vision Embedding
            v_embs = []
            for vid in batch_vids:
                chunks = self.vid_chunks.get(vid, [])
                if not chunks:
                    v_embs.append(torch.zeros(512).to(DEVICE))
                    continue
                frames = self.cache[self.idx_map[chunks[0]]].astype('float32')
                frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
                v_feat = torch.from_numpy(frames).mean(0).to(DEVICE)
                v_embs.append(v_feat)
            
            v_raw = torch.stack(v_embs)
            v_emb = self.vision_head(v_raw)
            
            # 3. Loss with Proximal Term
            loss = info_nce_loss(v_emb, t_emb)
            
            prox_loss = 0
            for p, p_g in zip(self.text_head.parameters(), t_head_global.parameters()):
                prox_loss += ((p - p_g)**2).sum()
            for p, p_g in zip(self.vision_head.parameters(), v_head_global.parameters()):
                prox_loss += ((p - p_g)**2).sum()
                
            total_batch_loss = loss + mu * prox_loss
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
            pbar.set_postfix({"loss": f"{total_batch_loss.item():.4f}"})

        avg_loss = total_loss / (len(self.vids) / BATCH_SIZE)
        return self.get_parameters(config={}), len(self.vids), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Evaluate Local Precision
        return 0.0, len(self.vids), {"accuracy": 0.0}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    args = parser.parse_args()
    
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=VCMRClient(args.client_id))

if __name__ == "__main__":
    main()
