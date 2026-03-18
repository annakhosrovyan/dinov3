from typing import Any, Tuple, Optional
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import math

from dinov3.data.datasets.channel_utils import to_five_channels
from dinov3.data.datasets.satlas_datasets import Sen1Dataset, Sen2Dataset
from dinov3.data.datasets.hdf5_dataset import BENDataset, IntelinairDataset, Sen12MSDataset
from dinov3.data.datasets.maid_dataset import MAIDDataset


class MixedSatelliteDataset(Dataset):
    def __init__(
        self,
        sen1_data_path: Optional[str] = None,
        sen1_stats_dir: Optional[str] = None,
        sen1_weight: Optional[float] = 1.0,
        sen2a_data_path: Optional[str] = None,
        sen2a_stats_dir: Optional[str] = None,
        sen2a_weight: Optional[float] = 1.0,
        sen2b_data_path: Optional[str] = None,
        sen2b_stats_dir: Optional[str] = None,
        sen2b_weight: Optional[float] = 1.0,
        ben_data_path: Optional[str] = None,
        ben_weight: Optional[float] = 1.0,
        sen12ms_data_path: Optional[str] = None,
        sen12ms_weight: Optional[float] = 1.0,
        intelinair_data_path: Optional[str] = None,
        intelinair_weight: Optional[float] = 1.0,
        maid_data_path: Optional[str] = None,
        maid_weight: Optional[float] = 1.0,
        transform=None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        def identity_transform(x):
            return x

        def parse_weight(value: Optional[float], dataset_name: str) -> float:
            if value is None:
                parsed = 1.0
            elif isinstance(value, str):
                parsed = float(value)
            else:
                parsed = float(value)
            if not math.isfinite(parsed) or parsed <= 0.0:
                raise ValueError(f"{dataset_name}_weight must be a positive finite value, got {value}")
            return parsed

        def add_dataset(dataset_name: str, dataset_obj: Dataset, weight_value: Optional[float]) -> None:
            raw_size = len(dataset_obj)
            if raw_size <= 0:
                raise ValueError(f"{dataset_name} dataset is empty")
            weight = parse_weight(weight_value, dataset_name)
            effective_size = max(1, int(raw_size * weight))
            self.datasets.append(dataset_obj)
            self.dataset_names.append(dataset_name)
            self.dataset_raw_sizes.append(raw_size)
            self.dataset_effective_sizes.append(effective_size)
            print(
                f"Added {dataset_name}Dataset: raw={raw_size} weight={weight:.4f} effective={effective_size}"
            )

        def add_sen2_dataset(dataset_name: str, data_path: str, stats_dir: str, weight_value: Optional[float]) -> None:
            sen2_ds = Sen2Dataset(
                data_path=data_path,
                stats_dir=stats_dir,
                transform=identity_transform,
                **kwargs,
            )
            add_dataset(dataset_name, sen2_ds, weight_value)

        self.datasets = []
        self.dataset_names = []
        self.dataset_raw_sizes = []
        self.dataset_effective_sizes = []

        if sen1_data_path is not None and sen1_stats_dir is not None:
            sen1_ds = Sen1Dataset(
                data_path=sen1_data_path,
                stats_dir=sen1_stats_dir,
                transform=identity_transform,
                **kwargs,
            )
            add_dataset("Sen1", sen1_ds, sen1_weight)

        if sen2a_data_path is not None and sen2a_stats_dir is not None:
            add_sen2_dataset("Sen2a", sen2a_data_path, sen2a_stats_dir, sen2a_weight)

        if sen2b_data_path is not None and sen2b_stats_dir is not None:
            add_sen2_dataset("Sen2b", sen2b_data_path, sen2b_stats_dir, sen2b_weight)

        if ben_data_path is not None:
            ben_ds = BENDataset(
                data_path=ben_data_path,
                transform=identity_transform,
                **kwargs,
            )
            add_dataset("BEN", ben_ds, ben_weight)

        if sen12ms_data_path is not None:
            sen12ms_ds = Sen12MSDataset(
                data_path=sen12ms_data_path,
                transform=identity_transform,
                **kwargs,
            )
            add_dataset("Sen12MS", sen12ms_ds, sen12ms_weight)

        if intelinair_data_path is not None:
            intelinair_ds = IntelinairDataset(
                data_path=intelinair_data_path,
                transform=identity_transform,
                **kwargs,
            )
            add_dataset("Intelinair", intelinair_ds, intelinair_weight)

        if maid_data_path is not None:
            maid_ds = MAIDDataset(
                data_path=maid_data_path,
                transform=identity_transform,
                **kwargs,
            )
            add_dataset("MAID", maid_ds, maid_weight)

        if not self.datasets:
            raise ValueError("At least one dataset must be specified")

        self.cumulative_sizes = [0]
        for effective_size in self.dataset_effective_sizes:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + effective_size)

        self.transform = transform
        print(f"Total weighted samples: {self.cumulative_sizes[-1]}")

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        for i, (start, end) in enumerate(zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])):
            if start <= index < end:
                dataset_idx = index - start
                base_size = self.dataset_raw_sizes[i]
                base_idx = dataset_idx % base_size
                images, band_names, data_path = self.datasets[i][base_idx]
                dataset_name = self.dataset_names[i]

                if isinstance(images, np.ndarray):
                    images = to_five_channels(images)

                if self.transform is not None:
                    images = self.transform(images, dataset_name=dataset_name)

                return images, ()

        raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")
