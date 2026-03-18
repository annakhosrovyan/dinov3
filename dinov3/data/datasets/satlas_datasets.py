import math
import os
import random
import hashlib
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from dinov3.data.datasets.channel_utils import validate_channel_count

class SatlasDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform,
        patch_size: int = 16,
        pred_ratio=0.5,
        pred_ratio_var=0.0,
        pred_aspect_ratio: Tuple[float, float] = (0.3, 1 / 0.3),
        pred_shape: str = "block",
        pred_start_epoch: int = 0,
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
        self.data_path = data_path
        self.transform = transform

    def load_stats(self, stats_dir: str, band: str) -> dict:
        if band == "tci":
            stats_path = os.path.join(stats_dir, "final_rgb_stats.npy")
        else:
            stats_path = os.path.join(stats_dir, f"{band}_band_stats.npy")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(stats_path)
        return np.load(stats_path, allow_pickle=True).item()

    def save_error_path(self, error_path: str) -> None:
        save_path = os.path.join(os.path.dirname(self.data_path), f"{os.path.basename(self.data_path)}_error_paths.txt")
        with open(save_path, "a") as f:
            f.write(f"{error_path}\n")

    def _manifest_candidates(self, filename: str) -> List[str]:
        data_manifest = os.path.join(self.data_path, filename)
        digest = hashlib.sha1(self.data_path.encode("utf-8")).hexdigest()[:16]
        base_name = os.path.basename(self.data_path.rstrip(os.sep)) or "dataset"
        cache_file = f"{base_name}_{digest}_{filename}"
        home_cache = os.path.join(os.path.expanduser("~"), ".cache", "dinov3_manifests", cache_file)
        tmp_cache = os.path.join(tempfile.gettempdir(), "dinov3_manifests", cache_file)
        return [data_manifest, home_cache, tmp_cache]

    def _save_manifest(self, manifest_paths: List[str], values: Dict[str, List[str]]) -> None:
        arrays = {key: np.asarray(paths, dtype=np.str_) for key, paths in values.items()}
        for manifest_path in manifest_paths:
            try:
                os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                np.savez_compressed(manifest_path, **arrays)
                return
            except OSError:
                continue

    def _load_manifest(self, manifest_paths: List[str], required_keys: Tuple[str, ...]) -> Optional[Dict[str, List[str]]]:
        for manifest_path in manifest_paths:
            if not os.path.exists(manifest_path):
                continue
            try:
                loaded = np.load(manifest_path, allow_pickle=False)
            except Exception:
                continue
            try:
                manifest: Dict[str, List[str]] = {}
                for key in required_keys:
                    if key not in loaded:
                        manifest = {}
                        break
                    values = loaded[key].tolist()
                    if isinstance(values, str):
                        manifest[key] = [values]
                    else:
                        manifest[key] = [str(value) for value in values]
                if manifest:
                    return manifest
            finally:
                loaded.close()
        return None


class Sen1Dataset(SatlasDataset):
    def __init__(
        self,
        data_path: str = "/nfs/h100/raid/rs/satlas_dataset/sentinel1",
        stats_dir: str = "/nfs/h100/raid/rs/satlas_dataset/stats/sentinel1_stats",
        transform=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_path=data_path, transform=transform, **kwargs)
        self.band_names = ["VV", "VH"]
        vv_stats = np.load(os.path.join(stats_dir, "vv_band_stats.npy"), allow_pickle=True).item()
        vh_stats = np.load(os.path.join(stats_dir, "vh_band_stats.npy"), allow_pickle=True).item()
        self.vv_mean = vv_stats["mean"]
        self.vv_std = vv_stats["std"]
        self.vh_mean = vh_stats["mean"]
        self.vh_std = vh_stats["std"]
        self.band_paths: Dict[str, List[str]] = {"VV": [], "VH": []}
        manifest_paths = self._manifest_candidates(".sen1_manifest_v1.npz")
        manifest = self._load_manifest(manifest_paths, ("VV", "VH"))
        if manifest is not None:
            self.band_paths["VV"] = manifest["VV"]
            self.band_paths["VH"] = manifest["VH"]
        else:
            subfolders = [
                os.path.join(data_path, d)
                for d in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, d))
            ]
            for subfolder in tqdm(subfolders, desc="SEN1 subfolders", unit="folder", mininterval=10):
                vv_folder = os.path.join(subfolder, "vv")
                vh_folder = os.path.join(subfolder, "vh")
                if not (os.path.isdir(vv_folder) and os.path.isdir(vh_folder)):
                    continue
                vv_pngs = {name for name in os.listdir(vv_folder) if name.lower().endswith(".png")}
                vh_pngs = {name for name in os.listdir(vh_folder) if name.lower().endswith(".png")}
                common_pngs = sorted(vv_pngs.intersection(vh_pngs))
                for png_name in common_pngs:
                    self.band_paths["VV"].append(os.path.join(vv_folder, png_name))
                    self.band_paths["VH"].append(os.path.join(vh_folder, png_name))
            self._save_manifest(manifest_paths, self.band_paths)
        if len(self.band_paths["VV"]) != len(self.band_paths["VH"]):
            raise ValueError("SEN1 manifest contains mismatched VV/VH lengths")
        self.ok_indices = list(range(len(self.band_paths["VV"])))

    def __len__(self) -> int:
        return len(self.band_paths["VV"])

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        if index not in self.ok_indices:
            index = int(np.random.choice(self.ok_indices))
        vv_path = self.band_paths["VV"][index]
        vh_path = self.band_paths["VH"][index]
        try:
            vv_img = cv2.imread(vv_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            vh_img = cv2.imread(vh_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        except Exception:
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        if vv_img is None or vh_img is None:
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        if vv_img.shape != vh_img.shape:
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        if np.isnan(vv_img).any() or np.isnan(vh_img).any():
            self.save_error_path(vv_path)
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        vv_img = (vv_img - self.vv_mean) / self.vv_std
        vh_img = (vh_img - self.vh_mean) / self.vh_std
        images = np.stack((vv_img, vh_img), axis=-1)
        images = images.astype(np.float32)
        validate_channel_count(images)
        images = self.transform(images)
        return images, self.band_names, self.data_path


class Sen2Dataset(SatlasDataset):
    def __init__(
        self,
        data_path: str,
        stats_dir: str,
        transform=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_path=data_path, transform=transform, **kwargs)
        self.band_names = ["R", "G", "B"]
        tci_folder = "tci"
        tci_stats = self.load_stats(stats_dir, "tci")
        self.means: List[float] = []
        self.stds: List[float] = []
        for band in ["R", "G", "B"]:
            self.means.append(tci_stats[band]["mean"])
            self.stds.append(tci_stats[band]["std"])
        self.band_paths: defaultdict = defaultdict(list)
        manifest_paths = self._manifest_candidates(".sen2_manifest_v1.npz")
        manifest = self._load_manifest(manifest_paths, (tci_folder,))
        if manifest is not None:
            self.band_paths[tci_folder] = manifest[tci_folder]
        else:
            subfolders = [
                os.path.join(data_path, d)
                for d in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, d))
            ]
            for subfolder in tqdm(subfolders, desc="SEN2 subfolders", unit="folder", mininterval=10):
                tci_path = os.path.join(subfolder, tci_folder)
                if not os.path.isdir(tci_path):
                    continue
                png_names = [name for name in os.listdir(tci_path) if name.lower().endswith(".png")]
                self.band_paths[tci_folder].extend(os.path.join(tci_path, name) for name in png_names)
            self._save_manifest(manifest_paths, {tci_folder: self.band_paths[tci_folder]})
        self.ok_indices = list(range(len(self.band_paths[tci_folder])))

    def __len__(self) -> int:
        return len(self.band_paths["tci"])

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        if index not in self.ok_indices:
            index = int(np.random.choice(self.ok_indices))
        channels: List[np.ndarray] = []
        try:
            tci_img = cv2.imread(self.band_paths["tci"][index])
        except Exception:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        if tci_img is None or tci_img.ndim < 3 or tci_img.shape[2] < 3:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        channels.append(tci_img[..., 2])
        channels.append(tci_img[..., 1])
        channels.append(tci_img[..., 0])
        shape_set = {ch.shape for ch in channels}
        if len(shape_set) != 1:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        combined = np.stack(channels, axis=-1)
        if np.isnan(combined).any():
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        combined = (combined - self.means) / self.stds
        images = combined.astype(np.float32)
        validate_channel_count(images)
        images = self.transform(images)
        return images, self.band_names, self.data_path


class NaipDataset(SatlasDataset):
    def __init__(
        self,
        data_path: str = "/nfs/h100/raid/rs/satlas_dataset/naip",
        stats_dir: str = "/nfs/h100/raid/rs/satlas_dataset/stats/naip_stats",
        transform=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_path=data_path, transform=transform, **kwargs)
        self.band_names = ["R", "G", "B"]
        tci_folder = "tci"
        tci_stats = self.load_stats(stats_dir, "tci")
        self.means = []
        self.stds = []
        for band in ["R", "G", "B"]:
            self.means.append(tci_stats[band]["mean"])
            self.stds.append(tci_stats[band]["std"])
        self.band_paths: defaultdict = defaultdict(list)
        manifest_paths = self._manifest_candidates(".naip_manifest_v1.npz")
        manifest = self._load_manifest(manifest_paths, (tci_folder,))
        if manifest is not None:
            self.band_paths[tci_folder] = manifest[tci_folder]
        else:
            subfolders = [
                os.path.join(data_path, d)
                for d in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, d))
            ]
            for subfolder in tqdm(subfolders, desc="NAIP subfolders", unit="folder", mininterval=10):
                tci_path = os.path.join(subfolder, tci_folder)
                if not os.path.isdir(tci_path):
                    continue
                png_names = [name for name in os.listdir(tci_path) if name.lower().endswith(".png")]
                self.band_paths[tci_folder].extend(os.path.join(tci_path, name) for name in png_names)
            self._save_manifest(manifest_paths, {tci_folder: self.band_paths[tci_folder]})
        self.ok_indices = list(range(len(self.band_paths[tci_folder])))

    def __len__(self) -> int:
        return len(self.band_paths["tci"])

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        if index not in self.ok_indices:
            index = int(np.random.choice(self.ok_indices))
        channels: List[np.ndarray] = []
        try:
            tci_img = cv2.imread(self.band_paths["tci"][index])
        except Exception:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        if tci_img is None or tci_img.ndim < 3 or tci_img.shape[2] < 3:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        channels.append(tci_img[..., 2])
        channels.append(tci_img[..., 1])
        channels.append(tci_img[..., 0])
        shape_set = {ch.shape for ch in channels}
        if len(shape_set) != 1:
            self.save_error_path(self.band_paths["tci"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        combined = np.stack(channels, axis=-1)
        if np.isnan(combined).any():
            self.save_error_path(self.band_paths["ir"][index])
            self.ok_indices.remove(index)
            new_index = int(np.random.choice(self.ok_indices))
            return self.__getitem__(new_index)
        combined = (combined - self.means) / self.stds
        images = combined.astype(np.float32)
        images = self.transform(images)
        return images, self.band_names, self.data_path

