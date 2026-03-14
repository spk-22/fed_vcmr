import torch
import torch.nn as nn
from src.config import (
    DEVICE, TRANSFORMER_EMBED_DIM, 
    TRANSFORMER_NUM_HEADS, TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_CHECKPOINT_PATH
)

class CrossModalTransformer(nn.Module):
    def __init__(self, embed_dim=TRANSFORMER_EMBED_DIM, in_dim=512, num_heads=TRANSFORMER_NUM_HEADS, num_layers=TRANSFORMER_NUM_LAYERS):
        super().__init__()
        # Input projections (512 to 256)
        self.vis_proj = nn.Linear(in_dim, embed_dim)
        self.txt_proj = nn.Linear(in_dim, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Grounding Head (predict start and end boundary percentiles)
        self.boundary_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2), # 2 outputs: Start Prob, End Prob
            nn.Sigmoid()
        )
        
    def forward(self, vis_feats, text_feat):
        # vis_feats: (Batch, Seq=8, 512)
        # text_feat: (Batch, Seq=1, 512)
        
        v = self.vis_proj(vis_feats) # (B, 8, 256)
        t = self.txt_proj(text_feat) # (B, 1, 256)
        
        # Sequence: [TXT, V1, V2, ... V8]
        seq = torch.cat([t, v], dim=1) # (B, 9, 256)
        
        # Transformer pass
        out_seq = self.transformer(seq) # (B, 9, 256)
        
        # Pool visual tokens
        vis_out = out_seq[:, 1:, :] # (B, 8, 256)
        chunk_rep = vis_out.mean(dim=1) # (B, 256)
        
        # Predict start/end adjustments (0.0 to 1.0 mapping across the chunk temporal range)
        boundaries = self.boundary_head(chunk_rep) # (B, 2)
        
        return boundaries

class TransformerInference:
    def __init__(self, checkpoint_path=TRANSFORMER_CHECKPOINT_PATH):
        self.device = DEVICE
        self.model = CrossModalTransformer().to(self.device)
        self.is_loaded = False
        
        if checkpoint_path.exists():
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.model.eval()
                self.is_loaded = True
                print(f"Transformer model loaded from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading Transformer checkpoint: {e}")
        else:
            print(f"Warning: Transformer checkpoint not found at {checkpoint_path}. Running with random weights.")

    @torch.no_grad()
    def predict_boundaries(self, vis_feats, text_feat):
        """
        vis_feats: (Batch, 8, 512) torch tensor
        text_feat: (Batch, 1, 512) torch tensor
        Returns: (Batch, 2) numpy array of [start_perc, end_perc]
        """
        vis_feats = vis_feats.to(self.device)
        text_feat = text_feat.to(self.device)
        
        output = self.model(vis_feats, text_feat)
        return output.cpu().numpy()

if __name__ == "__main__":
    # Smoke test
    infer = TransformerInference()
    v = torch.randn(1, 8, 512)
    t = torch.randn(1, 1, 512)
    out = infer.predict_boundaries(v, t)
    print(f"Inference output shape: {out.shape}, values: {out}")
