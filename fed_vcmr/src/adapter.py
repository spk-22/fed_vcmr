import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import DEVICE

class AdapterHead(nn.Module):
    """
    Lightweight Adapter module for fine-tuning on local data.
    Architecture: Bottleneck MLP with Residual Connection.
    """
    def __init__(self, input_dim=512, hidden_dim=128):
        super(AdapterHead, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input embeddings (batch_size, input_dim)
        Returns:
            torch.Tensor: Adapted and normalized embeddings (batch_size, input_dim)
        """
        residual = x

        # Bottleneck path
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)

        # Residual connection (scaled)
        out = residual + 0.1 * out

        # L2 normalization
        return F.normalize(out, p=2, dim=-1)

class AdapterManager:
    """
    Manages client-specific AdapterHeads for Federated Learning.
    Each client/domain gets its own independent AdapterHead instance.
    """
    def __init__(self):
        self.adapters = {} # Dict[str, AdapterHead]

    def get_adapter(self, client_id: str) -> AdapterHead:
        """
        Retrieves an existing adapter or creates a new one for the given client.
        Ensures the adapter is on the correct DEVICE.
        """
        if client_id not in self.adapters:
            # print(f"Initializing new AdapterHead for client: {client_id}")
            adapter = AdapterHead().to(DEVICE)
            self.adapters[client_id] = adapter

        return self.adapters[client_id]

    def set_mode(self, mode: str):
        """
        Sets the training mode for all managed adapters.
        Args:
            mode (str): 'train' or 'eval'
        """
        for adapter in self.adapters.values():
            if mode == 'train':
                adapter.train()
            elif mode == 'eval':
                adapter.eval()
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'eval'.")

# Helper function for safe integration
_ADAPTER_MANAGER = None

def get_adapter_manager():
    """Singleton loader for AdapterManager."""
    global _ADAPTER_MANAGER
    if _ADAPTER_MANAGER is None:
        _ADAPTER_MANAGER = AdapterManager()
    return _ADAPTER_MANAGER

def apply_adapter(tensor, client_id="client_1"):
    """
    Apply client-specific adapter if enabled.
    Default client_id="client_1" for simulation.
    """
    from src.config import USE_ADAPTER
    if USE_ADAPTER:
        manager = get_adapter_manager()
        adapter = manager.get_adapter(client_id)
        # Ensure adapter is in eval mode for inference by default
        if not adapter.training:
             adapter.eval()
        return adapter(tensor)
    return tensor
