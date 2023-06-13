import os
import random
import torch
import numpy as np

np.seterr(all="raise")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def standardize(image: np.ndarray, max_z: float = 3, min_max: tuple = (0, 1)) -> np.ndarray:
    img_mean = image.mean(axis=1)[:, None]
    img_std = image.std(axis=1)[:, None]
    image = (image - img_mean) / (img_std + 0.0000001)

    image = (np.clip(image, -max_z, max_z) + max_z) / (2 * max_z)
    image = image * (min_max[1] - min_max[0]) + min_max[0]
    return image


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    img_min = image.min(axis=1)[:, None]
    img_max = image.max(axis=1)[:, None]
    return (image - img_min) / (img_max - img_min)
