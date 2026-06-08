"""
Export a TorchScript FL trainer module for on-device training.
All heavy ops (matmul, softmax, normalize) use native BLAS on ARM64.
No autograd needed — gradients computed analytically.

Usage: python scripts/export_fl_trainer.py
Output: checkpoints/export/fl_trainer.pt
"""
import torch
import torch.nn.functional as F
from typing import Tuple
import os


class FLTrainer(torch.nn.Module):
    """
    TorchScript-compatible FL training module.
    Performs one training step: forward + analytical gradient + SGD update.
    All operations are torch.mm / F.softmax / F.normalize = native ARM BLAS.
    """

    def __init__(self):
        super().__init__()

    @torch.jit.export
    def project(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Linear projection + L2 normalize: y = normalize(x @ W^T)"""
        return F.normalize(torch.mm(x, W.t()), dim=-1)

    @torch.jit.export
    def train_step(
        self,
        text_raw: torch.Tensor,       # (bs, 512)
        vision_raw: torch.Tensor,      # (bs, 512)
        hard_vision_raw: torch.Tensor, # (bs * K, 512)
        text_W: torch.Tensor,          # (512, 512)
        vision_W: torch.Tensor,        # (512, 512)
        codebook: torch.Tensor,        # (M, 512) - NC5 Moment Codebook
        text_W_global: torch.Tensor,
        vision_W_global: torch.Tensor,
        lr: float,
        temp: float,
        mu: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One training step with NC5 Federated Moment Codebook Learning.
        Returns (loss, updated_text_W, updated_vision_W, updated_codebook).
        """
        bs = text_raw.size(0)
        num_hard = hard_vision_raw.size(0)
        K = num_hard // bs
        M = codebook.size(0)

        # 1. Forward: project through heads
        t_proj = F.normalize(torch.mm(text_raw, text_W.t()), dim=-1)   # (bs, 512)
        v_proj = F.normalize(torch.mm(vision_raw, vision_W.t()), dim=-1)  # (bs, 512)
        h_proj = F.normalize(torch.mm(hard_vision_raw, vision_W.t()), dim=-1) # (bs*K, 512)

        # 2. Retrieval Loss (InfoNCE)
        v_all = torch.cat([v_proj, h_proj], dim=0) # (bs + bs*K, 512)
        sims = torch.mm(t_proj, v_all.t()) / temp
        sm_row = F.softmax(sims, dim=1)
        diag = torch.diag(sm_row[:, :bs])
        retrieval_loss = -torch.log(diag + 1e-8).mean()

        # 3. NC5: Codebook Alignment Loss (Vector Quantization style)
        # Find nearest codebook entry for each vision feature (v_proj)
        cb_norm = F.normalize(codebook, dim=-1)
        cb_sims = torch.mm(v_proj, cb_norm.t()) # (bs, M)
        best_cb_indices = torch.argmax(cb_sims, dim=1) # (bs,)
        
        gathered_cb = torch.index_select(cb_norm, 0, best_cb_indices) # (bs, 512)
        codebook_loss = (1.0 - torch.sum(v_proj * gathered_cb, dim=1)).mean()

        total_loss = retrieval_loss + 0.1 * codebook_loss # 0.1 is lambda_cb

        # 4. Gradients
        eye_ext = torch.zeros_like(sims)
        eye_ext[:, :bs] = torch.eye(bs, device=sims.device)
        dSims = (sm_row - eye_ext) / (bs * temp)

        dV_cb = -0.1 * gathered_cb / bs # (bs, 512)

        dT = torch.mm(dSims, v_all)      # (bs, 512)
        dV_all = torch.mm(dSims.t(), t_proj) # (bs + bs*K, 512)
        dV = dV_all[:bs] + dV_cb
        dH = dV_all[bs:]

        dCB = torch.zeros_like(codebook)
        for i in range(bs):
            idx = best_cb_indices[i]
            dCB[idx] = dCB[idx] - (0.1 / bs) * v_proj[i]

        text_grad = torch.mm(dT.t(), text_raw)
        vision_grad = torch.mm(dV.t(), vision_raw) + torch.mm(dH.t(), hard_vision_raw)

        text_grad = text_grad + 2.0 * mu * (text_W - text_W_global)
        vision_grad = vision_grad + 2.0 * mu * (vision_W - vision_W_global)

        # 5. Update
        text_W_new = text_W - lr * text_grad
        vision_W_new = vision_W - lr * vision_grad
        codebook_new = codebook - lr * dCB

        return total_loss, text_W_new, vision_W_new, codebook_new

    def forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Default forward = project"""
        return self.project(x, W)


def main():
    model = FLTrainer()
    scripted = torch.jit.script(model)

    # Verify it works
    bs = 16
    K = 4
    M = 16
    text_raw = torch.randn(bs, 512)
    vision_raw = torch.randn(bs, 512)
    hard_vision_raw = torch.randn(bs * K, 512)
    text_W = torch.eye(512)
    vision_W = torch.eye(512)
    codebook = torch.randn(M, 512)

    loss, t_new, v_new, cb_new = scripted.train_step(
        text_raw, vision_raw, hard_vision_raw, text_W, vision_W, codebook, 
        text_W.clone(), vision_W.clone(),
        1e-5, 0.07, 0.01
    )
    print(f"Test train_step (NC5): loss={loss.item():.4f}, CB delta={torch.norm(cb_new - codebook).item():.6f}")

    # Save
    out_dir = "checkpoints/export"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fl_trainer.pt")
    scripted.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Size: {os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
