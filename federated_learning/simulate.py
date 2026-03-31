"""
federated_learning/simulate.py
Single-process FL simulation using flwr.simulation.
Replaces the server.py + 4x client launch pattern entirely.
"""
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import numpy as np
import os
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLIENTS = 4
NUM_ROUNDS = 5

# ── Model ───────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)

    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

def info_nce_loss(v, t, temp=0.07):
    # Cross-modal alignment loss
    sims = (v @ t.T) / temp
    labels = torch.arange(v.size(0)).to(v.device)
    loss_v = F.cross_entropy(sims, labels)
    loss_t = F.cross_entropy(sims.T, labels)
    return (loss_v + loss_t) / 2

# ── Shared Data Loaders (Lazy Loading) ──────────────────────────
_TEXT_FEATURES = None
_FRAME_CACHE = None
_IDX_MAP = None
_VID_CHUNKS = None

def get_text_features():
    global _TEXT_FEATURES
    if _TEXT_FEATURES is None:
        print("Loading precomputed text features...")
        _TEXT_FEATURES = torch.load('cache/fl_text_features.pt', map_location='cpu')
    return _TEXT_FEATURES

def get_frame_cache():
    global _FRAME_CACHE, _IDX_MAP, _VID_CHUNKS
    if _FRAME_CACHE is None:
        conn = sqlite3.connect('fedvcmr.db')
        n_total = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        all_chunks = conn.execute('SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
        
        _FRAME_CACHE = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(n_total, 8, 512))
        _IDX_MAP = {cid: i for i, (cid, _) in enumerate(all_chunks)}
        _VID_CHUNKS = {}
        for cid, vid in all_chunks:
            _VID_CHUNKS.setdefault(vid, []).append(cid)
        conn.close()
        print(f"Frame cache ready: {n_total} chunks.")
    return _FRAME_CACHE, _IDX_MAP, _VID_CHUNKS

# ── Flower Client ───────────────────────────────────────────────
class VCMRClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.text_head = ProjectionHead().to(DEVICE)
        self.vision_head = ProjectionHead().to(DEVICE)

        conn = sqlite3.connect('fedvcmr.db')
        self.vids = [r[0] for r in conn.execute(
            'SELECT video_id FROM fl_shards WHERE client_id = ?', (client_id,)
        ).fetchall()]
        conn.close()
        print(f"  [Client {client_id}] Registered with {len(self.vids)} videos.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.text_head.state_dict().values()] + \
               [val.cpu().numpy() for val in self.vision_head.state_dict().values()]

    def set_parameters(self, parameters):
        mid = len(parameters) // 2
        t_keys = list(self.text_head.state_dict().keys())
        v_keys = list(self.vision_head.state_dict().keys())
        self.text_head.load_state_dict({k: torch.tensor(v) for k, v in zip(t_keys, parameters[:mid])})
        self.vision_head.load_state_dict({k: torch.tensor(v) for k, v in zip(v_keys, parameters[mid:])})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        text_features = get_text_features()
        frame_cache, idx_map, vid_chunks = get_frame_cache()

        optimizer = torch.optim.AdamW(list(self.text_head.parameters()) + list(self.vision_head.parameters()), lr=1e-4)
        self.text_head.train()
        self.vision_head.train()

        BATCH_SIZE = 64
        total_loss = 0
        n_batches = 0

        # Simulation: 1 Epoch
        for i in range(0, len(self.vids), BATCH_SIZE):
            batch_vids = self.vids[i : i + BATCH_SIZE]
            
            # 1. Text side - from precomputed cache
            t_raw = torch.stack([text_features.get(vid, torch.zeros(512)) for vid in batch_vids]).to(DEVICE)
            t_emb = self.text_head(t_raw)
            
            # 2. Vision side - mean pool from cache
            # 2. Vision side - max pool from cache
            v_list = []
            for vid in batch_vids:
                chunks = vid_chunks.get(vid, [])
                if not chunks:
                    v_list.append(torch.zeros(512, device=DEVICE))
                    continue
                frames = frame_cache[idx_map[chunks[0]]].astype('float32')
                # L2 normalize frames before max-pool
                frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
                v_feat = torch.from_numpy(frames).max(dim=0)[0].to(DEVICE)
                # Re-normalize pooled feature
                v_feat = F.normalize(v_feat, dim=-1)
                v_list.append(v_feat)
            
            v_emb = self.vision_head(torch.stack(v_list))
            
            # 3. Optimization
            loss = info_nce_loss(v_emb, t_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [Client {self.client_id}] Round Loss: {avg_loss:.4f}")
        return self.get_parameters(config={}), len(self.vids), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.vids), {"accuracy": 0.0}

from flwr.common import Context

def audit_evaluate_metrics_aggregation(metrics):
    return {"accuracy": 0.0}

def weighted_average(metrics):
    # This aggregates fit() metrics (loss)
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_loss = sum([num_examples * m["loss"] for num_examples, m in metrics])
    return {"loss": float(weighted_loss / total_examples)}

def client_fn(context: Context) -> fl.client.Client:
    # Use partition-id from context
    client_id = int(context.node_config["partition-id"])
    print(f"[Factory] Spawning Virtual Client ID: {client_id}")
    return VCMRClient(client_id).to_client()

# ── Strategy with Weight Saving ──────────────────────────────────
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_loss = float('inf')

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # We can compute the weighted loss from metrics here
            current_loss = weighted_average([(res.num_examples, res.metrics) for _, res in results])["loss"]
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"New Best Global Loss: {self.best_loss:.4f} at Round {server_round}. Saving...")
                
                params = fl.common.parameters_to_ndarrays(aggregated_parameters)
                mid = len(params) // 2
                t_head = ProjectionHead()
                v_head = ProjectionHead()
                
                state_dict = {
                    'text_head': {k: torch.tensor(v) for k, v in zip(t_head.state_dict().keys(), params[:mid])},
                    'vision_head': {k: torch.tensor(v) for k, v in zip(v_head.state_dict().keys(), params[mid:])}
                }
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(state_dict, 'checkpoints/fl_global_model.pt')
                
            # Periodic checkpoints for learning curve
            if server_round in [1, 10, 20, 30, 40, 50]:
                params = fl.common.parameters_to_ndarrays(aggregated_parameters)
                mid = len(params) // 2
                t_head = ProjectionHead()
                v_head = ProjectionHead()
                state_dict = {
                    'text_head': {k: torch.tensor(v) for k, v in zip(t_head.state_dict().keys(), params[:mid])},
                    'vision_head': {k: torch.tensor(v) for k, v in zip(v_head.state_dict().keys(), params[mid:])}
                }
                torch.save(state_dict, f'checkpoints/fl_global_round_{server_round}.pt')
                print(f"Periodic Checkpoint: Round {server_round} saved.")

            if server_round == NUM_ROUNDS:
                print(f"Final Round {server_round} complete. Best Loss: {self.best_loss:.4f}")

        return aggregated_parameters, aggregated_metrics

# ── Simulation Runner ───────────────────────────────────────────
if __name__ == "__main__":
    NUM_ROUNDS = 50 
    print(f"Starting Federated Simulation (Virtual Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS})...")
    
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=weighted_average,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), 
        strategy=strategy,
        client_resources={"num_gpus": 0.25, "num_cpus": 2},
    )

    print("\n--- Federated Simulation Complete ---")
    print("Best Global Loss Achieved:", strategy.best_loss)
    print("Aggregate Losses (Fit):", history.metrics_distributed_fit)
