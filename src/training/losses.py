import torch
import torch.nn.functional as F

def infonce_loss(visual_emb, text_emb, temperature=0.07):
    """
    Symmetric InfoNCE loss for contrastive learning.
    visual_emb: (B, 256) L2-normalized visual features
    text_emb:   (B, 256) L2-normalized text features
    """
    # Similarity matrix (B x B)
    # Since inputs are normalized, matmul result is cosine similarity
    logits = torch.matmul(text_emb, visual_emb.T) / temperature

    # Ground truth: diagonal elements are positive pairs
    labels = torch.arange(len(logits), device=logits.device)

    # Symmetric loss (Text-to-Video and Video-to-Text)
    loss_t2v = F.cross_entropy(logits, labels)
    loss_v2t = F.cross_entropy(logits.T, labels)

    return (loss_t2v + loss_v2t) / 2
