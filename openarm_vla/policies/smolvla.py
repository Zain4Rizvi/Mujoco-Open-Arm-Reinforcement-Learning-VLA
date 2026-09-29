from __future__ import annotations

from pathlib import Path

import numpy as np

from openarm_vla.constants import ACTION_DIM, STATE_DIM


class SmolVLAAdapter:
    chunk_size = 16

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        self.checkpoint = checkpoint
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("SmolVLAAdapter requires `uv sync --extra train`. See README.md.") from e
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
        cfg = self.policy.config
        cfg.device = device
        if cfg.action_feature.shape[0] == ACTION_DIM:
            self.pre, self.post = make_pre_post_processors(
                cfg, str(checkpoint), preprocessor_overrides={"device_processor": {"device": device}}
            )
        else:
            # Base checkpoint (SO-100, 6-D): SmolVLA pads state/action to 32 dims, so our shapes fit the
            # same weights. Its 6-D norm stats don't apply; processors without stats pass values through.
            img = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            cfg.input_features = {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
                "observation.images.image_front": img,
                "observation.images.image_wrist": img,
            }
            cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
            self.pre, self.post = make_pre_post_processors(cfg)
        # Turing GPUs (GTX 16xx) have no bf16 compute.
        self.policy.float().to(device).eval()

    def predict_chunk(self, obs: dict) -> np.ndarray:
        import torch

        def img(a):
            return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0

        batch = self.pre(
            {
                "observation.images.image_front": img(obs["image_front"]),
                "observation.images.image_wrist": img(obs["image_wrist"]),
                "observation.state": torch.from_numpy(obs["state"]).float(),
                "task": obs["instruction"],
            }
        )
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(batch)
        actions = self.post(actions)
        return actions[0, : self.chunk_size].numpy().astype(np.float32)
