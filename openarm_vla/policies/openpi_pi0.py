from __future__ import annotations

from pathlib import Path


class OpenPiAdapter:
    """pi0 / openpi inference adapter. Requires `openpi` installed (see README)."""

    chunk_size = 16

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = checkpoint
        try:
            import openpi  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "OpenPiAdapter requires Physical Intelligence openpi. "
                "Install from https://github.com/Physical-Intelligence/openpi — see README.md."
            ) from e
        raise NotImplementedError(
            "Wire OpenPI policy loading here after `uv pip install` of openpi. "
            "Map cameras image_front/image_wrist, state dim 15, action dim 8, chunk 16."
        )
