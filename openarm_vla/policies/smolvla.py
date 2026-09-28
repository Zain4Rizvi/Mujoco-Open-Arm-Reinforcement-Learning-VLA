from __future__ import annotations

from pathlib import Path


class SmolVLAAdapter:
    chunk_size = 16

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        self.checkpoint = checkpoint
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("SmolVLAAdapter requires torch and lerobot. See README.md.") from e
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
        self.policy.eval()
        self.device = device
        self.policy.to(device)

    def predict_chunk(self, obs: dict):
        import torch
        import numpy as np

        batch = {
            "observation.images.image_front": torch.from_numpy(obs["image_front"]).permute(2, 0, 1).float()[None] / 255.0,
            "observation.images.image_wrist": torch.from_numpy(obs["image_wrist"]).permute(2, 0, 1).float()[None] / 255.0,
            "observation.state": torch.from_numpy(obs["state"]).float()[None],
            "task": [obs["instruction"]],
        }
        batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            action = self.policy.select_action(batch)
        a = action.detach().cpu().numpy()
        if a.ndim == 1:
            a = a[None, :]
        return a.astype(np.float32)
