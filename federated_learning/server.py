# federated_learning/server.py
"""
Flower (flwr) Server for Federated Learning.
Orchestrates weight aggregation using FedAvg.
"""
import flwr as fl
import torch
import numpy as np

def main():
    # 1. Define Strategy (Aggregator)
    # We use FedAvg for now, but can switch to FedProx or Trimmed Mean
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,          # All 4 clients must participate
        fraction_evaluate=1.0,     # All 4 clients must evaluate
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )

    # 2. Start Flower Server
    print("Starting Federated Learning Server on [::]:8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10), # Start with 10 for testing
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
