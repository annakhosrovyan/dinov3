import math
import os
import random
from typing import Any, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from dinov3.data.datasets.channel_utils import validate_channel_count

class MAIDDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform=None,
        patch_size: int = 224,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.band_names = ["R", "G", "B"]
        self.data_path = data_path
        self.transform = transform
        self.psz = patch_size
        exts = (".png", ".jpg", ".jpeg")
        img_paths: List[str] = []
        for dirpath, _, filenames in os.walk(data_path):
            for name in filenames:
                lower = name.lower()
                if lower.endswith(exts):
                    img_paths.append(os.path.join(dirpath, name))
        self.img_paths = img_paths
        self.data_len = len(self.img_paths)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        img_path = self.img_paths[index]
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"failed to read image {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        validate_channel_count(image)
        images = self.transform(image)
        for arr in images:
            if np.isnan(arr).any():
                raise RuntimeError(f"NaN in image {img_path}")
        return images, self.band_names, img_path

