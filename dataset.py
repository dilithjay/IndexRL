import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class SatelliteDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir):
        super().__init__()
        
        self.image_list = sorted(glob(os.path.join(image_dir, '*.npy')))
        self.mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.transform = ToTensor()
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = np.load(self.image_list[idx]).astype(float)
        mask = Image.open(self.mask_list[idx])
        
        return torch.tensor(image), self.transform(mask) / 255
