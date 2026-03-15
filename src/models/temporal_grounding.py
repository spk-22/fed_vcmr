import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalTransformer(nn.Module):
    def __init__(self, visual_dim=512, query_dim=512, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        
        # Projections to common hidden dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        
        # Positional Encoding for 8 visual tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Regressor Heads (Start, End)
        # We output 2 values: [start, end] normalized to [0, 1]
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid() 
        )

    def forward(self, visual_features, query_features):
        """
        visual_features: (batch, 8, visual_dim)
        query_features: (batch, query_dim)
        """
        # Project to hidden dim
        v = self.visual_proj(visual_features)  # (batch, 8, 256)
        q = self.query_proj(query_features).unsqueeze(1)  # (batch, 1, 256)
        
        # Add positional encoding to visual tokens
        v = v + self.pos_embed
        
        # Concatenate query as the first token (like a CLS token for grounding)
        # Sequence becomes [Q, V1, V2, ..., V8] -> length 9
        tokens = torch.cat([q, v], dim=1)  # (batch, 9, 256)
        
        # Pass through Transformer
        transformed = self.transformer(tokens)  # (batch, 9, 256)
        
        # Take the query token output for prediction
        # Alternatively, max-pool or average-pool. Query token is standard.
        query_out = transformed[:, 0, :]  # (batch, 256)
        
        # Predict [start, end]
        predictions = self.regressor(query_out)  # (batch, 2)
        
        return predictions

def temporal_iou_loss(pred, target):
    """
    pred, target: (batch, 2) with [start, end] in [0, 1]
    """
    pred_s, pred_e = pred[:, 0], pred[:, 1]
    target_s, target_e = target[:, 0], target[:, 1]
    
    # Ensure s < e (clamping for safety)
    pred_e = torch.max(pred_s + 0.01, pred_e)
    
    inter_s = torch.max(pred_s, target_s)
    inter_e = torch.min(pred_e, target_e)
    
    inter = torch.clamp(inter_e - inter_s, min=0)
    union = (pred_e - pred_s) + (target_e - target_s) - inter
    
    iou = inter / (union + 1e-6)
    return 1 - iou.mean()

class GroundingLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_iou=1.0):
        super().__init__()
        self.w_l1 = w_l1
        self.w_iou = w_iou
    
    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        iou_loss = temporal_iou_loss(pred, target)
        return self.w_l1 * l1_loss + self.w_iou * iou_loss
