import math
import os
import random
from typing import Any, List, Tuple

import h5py
import numpy as np
from torch.utils.data import Dataset

from dinov3.data.datasets.channel_utils import validate_channel_count

class HDF5Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform,
        patch_size: int,
        pred_ratio,
        pred_ratio_var,
        pred_aspect_ratio,
        pred_shape: str = "block",
        pred_start_epoch: int = 0,
        data_key: str = "BEN",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.psz = patch_size
        if isinstance(pred_ratio, list) and len(pred_ratio) == 1:
            self.pred_ratio = pred_ratio[0]
        else:
            self.pred_ratio = pred_ratio
        if isinstance(pred_ratio_var, list) and len(pred_ratio_var) == 1:
            self.pred_ratio_var = pred_ratio_var[0]
        else:
            self.pred_ratio_var = pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        self.hdf5 = h5py.File(data_path, "r")
        self.data = self.hdf5[data_key]
        self.data_key = data_key
        self.data_len = self.data.shape[0]
        self.means = self.hdf5["means"][:]
        self.stds = self.hdf5["stds"][:]
        band_names_bytes = self.hdf5["band_names"][:]
        self.band_names: List[str] = [x.decode("utf-8") for x in band_names_bytes]
        self.band_indices: List[int] = []
        self.transform = transform
        self.data_path = data_path
        self.ok_indices: List[int] = list(range(self.data.shape[0]))

    def __len__(self) -> int:
        return self.data_len

    def __del__(self) -> None:
        try:
            self.hdf5.close()
        except Exception:
            pass

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_pred_ratio(self) -> float:
        if hasattr(self, "epoch") and self.epoch < self.pred_start_epoch:
            return 0.0
        if isinstance(self.pred_ratio, list):
            pred_ratio_list: List[float] = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                if prv > 0:
                    pr = random.uniform(prm - prv, prm + prv)
                else:
                    pr = prm
                pred_ratio_list.append(pr)
            pred_ratio = random.choice(pred_ratio_list)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            if self.pred_ratio_var > 0:
                pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + self.pred_ratio_var)
            else:
                pred_ratio = self.pred_ratio
        return float(pred_ratio)

    def get_masks(self, images) -> List[np.ndarray]:
        masks: List[np.ndarray] = []
        for img in images:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except Exception:
                continue
            high = self.get_pred_ratio() * H * W
            if self.pred_shape == "block":
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count
                    delta = 0
                    for _ in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)
                            num_masked = mask[top : top + h, left : left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if not mask[i, j]:
                                            mask[i, j] = True
                                            delta += 1
                        if delta > 0:
                            break
                    if delta == 0:
                        break
                    mask_count += delta
            elif self.pred_shape == "rand":
                mask = np.hstack(
                    [
                        np.zeros(H * W - int(high)),
                        np.ones(int(high)),
                    ]
                ).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
            else:
                raise ValueError("Unsupported pred_shape")
            masks.append(mask)
        return masks

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        if index not in self.ok_indices:
            index = random.choice(self.ok_indices)
        if self.band_indices:
            orig_image = self.data[index][..., self.band_indices]
        else:
            orig_image = self.data[index]
        while np.isnan(orig_image).any():
            self.ok_indices.remove(index)
            index = random.choice(self.ok_indices)
            if self.band_indices:
                orig_image = self.data[index][..., self.band_indices]
            else:
                orig_image = self.data[index]
        if orig_image.dtype == np.uint8:
            orig_image = (orig_image - self.means) / self.stds
        orig_image = orig_image.astype(np.float32)
        validate_channel_count(orig_image)
        images = self.transform(orig_image)
        return images, self.band_names, self.data_path


class Sen12MSDataset(HDF5Dataset):
    def __init__(
        self,
        data_path: str = "/nfs/ap/mnt/frtn/rs-multiband/sen12ms.h5",
        transform=None,
        patch_size: int = 16,
        pred_ratio=0.5,
        pred_ratio_var=0.0,
        pred_aspect_ratio: Tuple[float, float] = (0.3, 1 / 0.3),
        pred_shape: str = "block",
        pred_start_epoch: int = 0,
        data_key: str = "SEN12MS",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_path=data_path,
            transform=transform,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch,
            data_key=data_key,
            **kwargs,
        )
        target_bands: List[str] = ["R", "G", "B", "VV", "VH"]
        band_to_idx = {name: idx for idx, name in enumerate(self.band_names)}
        self.band_indices = [band_to_idx[name] for name in target_bands]
        self.band_names = [self.band_names[idx] for idx in self.band_indices]
        self.means = self.means[self.band_indices]
        self.stds = self.stds[self.band_indices]


class BENDataset(HDF5Dataset):
    def __init__(
        self,
        data_path: str = "/nfs/ap/mnt/frtn/rs-multiband/BEN_complete.h5",
        transform=None,
        patch_size: int = 120,
        pred_ratio=0.5,
        pred_ratio_var=0.0,
        pred_aspect_ratio: Tuple[float, float] = (0.3, 1 / 0.3),
        pred_shape: str = "block",
        pred_start_epoch: int = 0,
        data_key: str = "BEN",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_path=data_path,
            transform=transform,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch,
            data_key=data_key,
            **kwargs,
        )
        target_bands: List[str] = ["R", "G", "B", "VV", "VH"]
        band_to_idx = {name: idx for idx, name in enumerate(self.band_names)}
        self.band_indices = [band_to_idx[name] for name in target_bands]
        self.band_names = [self.band_names[idx] for idx in self.band_indices]

class IntelinairDataset(HDF5Dataset):
    def __init__(
        self,
        data_path: str = "/nfs/ap/mnt/frtn/rs-multiband/intelinair.h5",
        transform=None,
        patch_size: int = 320,
        pred_ratio=0.5,
        pred_ratio_var=0.0,
        pred_aspect_ratio: Tuple[float, float] = (0.3, 1 / 0.3),
        pred_shape: str = "block",
        pred_start_epoch: int = 0,
        data_key: str = "intelinair",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_path=data_path,
            transform=transform,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch,
            data_key=data_key,
            **kwargs,
        )
        target_bands: List[str] = ["R", "G", "B"]
        band_to_idx = {name: idx for idx, name in enumerate(self.band_names)}
        self.band_indices = [band_to_idx[name] for name in target_bands]
        self.band_names = [self.band_names[idx] for idx in self.band_indices]
        self.means = self.means[self.band_indices]
        self.stds = self.stds[self.band_indices]

