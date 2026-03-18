import numpy as np


def validate_channel_count(image: np.ndarray) -> int:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image with 3 dimensions, got shape {image.shape}")
    channel_count = int(image.shape[-1])
    if channel_count not in (2, 3, 5):
        raise ValueError(f"Expected channel count in (2, 3, 5), got {channel_count}")
    return channel_count


def to_five_channels(image: np.ndarray) -> np.ndarray:
    channel_count = validate_channel_count(image)
    if channel_count == 5:
        return image

    height, width = image.shape[:2]
    if channel_count == 3:
        pad = np.zeros((height, width, 2), dtype=image.dtype)
        return np.concatenate([image, pad], axis=-1)

    pad = np.zeros((height, width, 3), dtype=image.dtype)
    return np.concatenate([pad, image], axis=-1)
