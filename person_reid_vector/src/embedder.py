from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


class ReIDEmbedder:
    """OSNet-based person Re-ID feature extractor.

    The model is frozen and used only to generate appearance embeddings.
    """

    def __init__(self, model_name: str = "osnet_x1_0", device: str = "auto"):
        import torchreid

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True,
        )
        self.model.eval().to(self.device)
        self.transform = torchreid.utils.transforms.build_transforms(
            height=256, width=128, is_train=False
        )[1]

    @torch.inference_mode()
    def embed_images(self, paths: Iterable[str | Path], batch_size: int = 32) -> np.ndarray:
        paths = list(paths)
        if not paths:
            return np.empty((0, 512), dtype=np.float32)

        all_embeddings = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            tensors = []
            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                tensors.append(self.transform(image))
            batch = torch.stack(tensors).to(self.device)
            features = self.model(batch)
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def embed_image(self, path: str | Path) -> np.ndarray:
        return self.embed_images([path])[0]
