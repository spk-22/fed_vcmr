import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import os
import sys
import copy
from typing import List, Dict
from pathlib import Path

# Ensure src can be imported (assuming src is in the same directory as this script)
sys.path.append(str(Path(__file__).parent))

import src.config
from src.config import DEVICE
from src.dataset import FedVCMRDataset
from src.model import ProjectionHead
from src.adapter import AdapterHead, get_adapter_manager
from src.backbone import MobileCLIPWrapper

class Client:
    def __init__(self, client_id: str, dataset, batch_size=32, lr=1e-4):
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr

        # 1. Local Adapter (Retrieve from Manager via Singleton)
        self.manager = get_adapter_manager()
        self.adapter = self.manager.get_adapter(client_id)

        # 2. Local Projection Head (independent instance per client)
        self.projection_head = ProjectionHead().to(DEVICE)

        # 3. Optimizer for both
        # Requirement: "projection gradients tracked" -> optimize both
        self.optimizer = optim.Adam(
            list(self.adapter.parameters()) + list(self.projection_head.parameters()),
            lr=self.lr
        )
        self.criterion = nn.CrossEntropyLoss()

    def set_projection_weights(self, global_weights: Dict):
        """Load global projection weights into local model."""
        # load_state_dict handles device placement if model is on GPU
        self.projection_head.load_state_dict(global_weights)

    def get_projection_weights(self) -> Dict:
        """Return local projection weights (state_dict). Moves to CPU to save GPU memory."""
        return {k: v.cpu() for k, v in self.projection_head.state_dict().items()}

    def train_local(self, epochs=1):
        print(f"Client {self.client_id}: Training local model ({epochs} epochs)...")
        self.adapter.train()
        self.projection_head.train()

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0

            for batch_idx, (video_emb, text_emb) in enumerate(loader):
                video_emb = video_emb.to(DEVICE).float()
                text_emb = text_emb.to(DEVICE).float()

                self.optimizer.zero_grad()

                # Forward Pass
                v_proj = self.projection_head(video_emb)
                t_proj = self.projection_head(text_emb)

                v_final = self.adapter(v_proj)
                t_final = self.adapter(t_proj)

                # Compute Loss (Symmetric Cross Entropy with Temperature)
                temperature = 0.05
                sim = torch.matmul(v_final, t_final.T) / temperature
                labels = torch.arange(video_emb.size(0), device=DEVICE)

                loss_v = self.criterion(sim, labels)
                loss_t = self.criterion(sim.T, labels)
                loss = (loss_v + loss_t) / 2

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / count if count > 0 else 0
            print(f"  > Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

class Server:
    def __init__(self):
        self.global_projection = ProjectionHead().to(DEVICE)
        print("Server: Global model initialized.")

    def distribute_model(self, clients: List[Client]):
        """Send global projection weights to each client."""
        print("Server: Distributing global model...")
        global_weights = self.global_projection.state_dict()
        for client in clients:
            # Deepcopy ensures clients have their own memory copy
            client.set_projection_weights(copy.deepcopy(global_weights))

    def aggregate(self, client_weights_list: List[Dict]):
        """Aggregate client weights (Simple Average)."""
        print("Server: Aggregating client updates...")
        if not client_weights_list:
            return

        # Initialize average weights with the first client's weights
        avg_weights = copy.deepcopy(client_weights_list[0])
        num_clients = len(client_weights_list)

        # Sum up weights
        for key in avg_weights.keys():
            for i in range(1, num_clients):
                avg_weights[key] += client_weights_list[i][key]

            # Divide by number of clients
            avg_weights[key] = torch.div(avg_weights[key], num_clients)

        # Update global model
        self.global_projection.load_state_dict(avg_weights)
        print("Server: Aggregation complete. Global model updated.")

def run_simulation(num_rounds=3, num_clients=2, epochs_per_round=1):
    print("=== Federated Learning Simulation Skeleton ===")

    # 0. Configuration
    src.config.USE_PROJECTION = True
    src.config.USE_ADAPTER = True

    # 1. Setup Data
    print("Initializing Backbone and Dataset...")
    # Using backbone on CPU for text embedding generation
    backbone = MobileCLIPWrapper(device=DEVICE)
    # Limit dataset size for simulation speed
    full_dataset = FedVCMRDataset(backbone=backbone, split='train', limit=50)

    # 1. NON-IID DATA SPLIT
    # Split dataset into disjoint chunks (consecutive indices) to simulate domain shifts.
    # We do NOT shuffle globally to preserve any inherent ordering (or simply force distinct sets).
    total_size = len(full_dataset)
    chunk_size = total_size // num_clients

    client_datasets = []
    start_idx = 0

    print("\nCreating Non-IID Data Splits:")
    for i in range(num_clients):
        # Determine index range for this client
        end_idx = start_idx + chunk_size
        if i == num_clients - 1:
            end_idx = total_size # Ensure we cover the remainder

        indices = list(range(start_idx, end_idx))
        subset = Subset(full_dataset, indices)
        client_datasets.append(subset)

        # 4. ADD DEBUG LOGS
        print(f"  > Client {i+1} -> indices [{start_idx} : {end_idx-1}] (Size: {len(subset)})")

        start_idx = end_idx

    # 2. Setup Server and Clients
    server = Server()
    clients = []

    for i in range(num_clients):
        client = Client(
            client_id=f"client_{i+1}",
            dataset=client_datasets[i],
            batch_size=8 # Small batch for small dataset
        )
        clients.append(client)
        print(f"Initialized Client {i+1} with {len(client_datasets[i])} samples.")

    # 3. Simulation Loop
    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1} / {num_rounds} ---")

        # A. Distribute
        server.distribute_model(clients)

        # B. Train Local
        collected_weights = []
        for client in clients:
            client.train_local(epochs=epochs_per_round)
            collected_weights.append(client.get_projection_weights())

        # C. Aggregate
        server.aggregate(collected_weights)

    print("\nSimulation Finished Successfully.")

if __name__ == "__main__":
    run_simulation()
