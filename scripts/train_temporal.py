import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import sys
import os

# Local imports
sys.path.insert(0, '.')
from src.models.temporal_grounding import CrossModalTransformer, GroundingLoss

# Config
DATA_PATH = 'cache/anet_data_consolidated.pt'
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = 'checkpoints/temporal_grounding_best.pt'

class AnetGroundingDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['features'], item['query_embed'], item['target']

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run scripts/package_anet_features.py first.")
        return

    dataset = AnetGroundingDataset(DATA_PATH)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    model = CrossModalTransformer().to(DEVICE)
    criterion = GroundingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for visual, query, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            visual, query, target = visual.to(DEVICE), query.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(visual, query)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for visual, query, target in val_loader:
                visual, query, target = visual.to(DEVICE), query.to(DEVICE), target.to(DEVICE)
                pred = model(visual, query)
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model to {SAVE_PATH}")

if __name__ == "__main__":
    train()
