import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from dinov3.data.datasets.channel_utils import to_five_channels



class HDF5Augmentation(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.flip_and_color_jitter = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ], p=0.5),
            # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        ])

        self.global_crops_number = global_crops_number
        
        # transformation for the first global crop
        self.global_transfo1 = A.Compose([
            A.RandomResizedCrop((global_crops_size, global_crops_size), scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0),
            ToTensorV2(),
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = A.Compose([
            A.RandomResizedCrop((global_crops_size, global_crops_size), scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.1),
            ToTensorV2(),
        ])
        
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = A.Compose([
            A.RandomResizedCrop((local_crops_size, local_crops_size), scale=local_crops_scale, interpolation=cv2.INTER_CUBIC),
            self.flip_and_color_jitter,
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.5),
            ToTensorV2(),
        ])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True
        
        assert isinstance(image, np.ndarray)
        image = to_five_channels(image)
        
        global_crops = []
        global_crops.append(self.global_transfo1(image=image)["image"])
        
        for _ in range(self.global_crops_number - 1):
            global_crops.append(self.global_transfo2(image=image)["image"])
        
        local_crops = []
        for _ in range(self.local_crops_number):
            local_crops.append(self.local_transfo(image=image)["image"])
        
        output["global_crops"] = global_crops
        output["global_crops_teacher"] = global_crops
        output["local_crops"] = local_crops
        output["offsets"] = ()
        
        return output