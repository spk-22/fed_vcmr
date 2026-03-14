# src/models/dgse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DGSE(nn.Module):
    """
    Dual-Granularity Segment Embedding.

    Coarse branch : static center-biased weighted average (no params)
    Fine branch   : query-conditioned cross-attention over N frames
    Fusion        : learned scalar α blends fine and coarse
    """

    def __init__(self, dim=512, n_frames=8):
        super().__init__()
        self.dim      = dim
        self.n_frames = n_frames

        # Query-conditioned attention
        # Projects query into attention logits over n_frames positions
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, n_frames)
        )

        # Learned fusion weight — sigmoid gives α in (0, 1)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

        # Center-biased static weights (same as current pipeline)
        w = torch.tensor(
            [0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5],
            dtype=torch.float32
        )
        self.register_buffer('coarse_weights', w / w.sum())

    def coarse(self, frames):
        """
        frames: (B, N, D) or (N, D)
        returns: (B, D) or (D,)
        """
        w = self.coarse_weights  # (N,)
        if frames.dim() == 2:
            return (frames * w.unsqueeze(-1)).sum(0)
        return (frames * w.unsqueeze(0).unsqueeze(-1)).sum(1)

    def fine(self, frames, query):
        """
        frames: (B, N, D)
        query:  (B, D)
        returns: (B, D)
        """
        # Compute attention weights conditioned on query
        logits  = self.attn_proj(query)          # (B, N)
        weights = F.softmax(logits, dim=-1)       # (B, N)
        # Weighted sum over frames
        return (frames * weights.unsqueeze(-1)).sum(1)  # (B, D)

    def forward(self, frames, query):
        """
        frames: (B, N, D) — raw frame features for a chunk
        query:  (B, D)    — normalized query embedding
        returns: (B, D)   — normalized dual-granularity embedding
        """
        c = self.coarse(frames)                   # (B, D)
        f = self.fine(frames, query)              # (B, D)

        alpha = torch.sigmoid(self.fusion_alpha)  # scalar in (0,1)
        out   = alpha * f + (1 - alpha) * c       # (B, D)
        return F.normalize(out, dim=-1)


class DGSERetriever:
    """
    Drop-in replacement for the current retrieval pipeline.
    Uses FAISS coarse index for candidate retrieval,
    then DGSE reranking for final ordering.
    """

    def __init__(self, faiss_index, chunk_metadata,
                 frame_cache, dgse_model,
                 vision_head, text_head,
                 device='cpu'):
        self.index    = faiss_index
        self.meta     = chunk_metadata   # list of (chunk_id, video_id, start, end)
        self.cache    = frame_cache      # np.memmap (N, 8, 512)
        self.dgse     = dgse_model
        self.vh       = vision_head
        self.th       = text_head
        self.device   = device

        w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5],
                     dtype='float32')
        self.w = w / w.sum()

    @torch.no_grad()
    def retrieve(self, query_emb, top_k=5, n_candidates=50):
        """
        query_emb: (1, 512) normalized query embedding
                   already passed through text_head
        Returns list of (chunk_id, video_id, start, end, score)
        """
        # Stage 1 — coarse FAISS retrieval
        q_np = query_emb.cpu().numpy().astype('float32')
        D, I = self.index.search(q_np, n_candidates)

        # Stage 2 — DGSE reranking on candidates
        candidates = []
        for rank, idx in enumerate(I[0]):
            if idx < 0:
                continue
            chunk_id, video_id, start, end = self.meta[idx]
            row    = idx
            frames = torch.tensor(
                self.cache[row].astype('float32')
            ).unsqueeze(0).to(self.device)          # (1, 8, 512)

            dgse_emb = self.dgse(frames, query_emb) # (1, 512)
            score    = (dgse_emb @ query_emb.T).item()
            candidates.append((chunk_id, video_id, start, end, score))

        # Sort by DGSE score
        candidates.sort(key=lambda x: x[4], reverse=True)
        return candidates[:top_k]
