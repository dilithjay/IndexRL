import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize


class SatelliteDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()

        self.image_list = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.mask_list = sorted(glob(os.path.join(data_dir, "*.png")))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = np.load(self.image_list[idx]).astype(float)
        mask = Image.open(self.mask_list[idx])

        image = torch.tensor(image)
        image = normalize(
            image,
            [455, 675, 400, 1000, 2480, 2905, 3040, 3130, 1810, 950],
            [185, 148, 99, 225, 465, 557, 625, 594, 412, 306],
        )
        mask = to_tensor(mask) / 255

        return image, mask
