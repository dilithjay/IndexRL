from glob import glob
import os
import random
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from expression_handler import check_unitless_validity, eval_expression


class CustomDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, exp: list, img_size: int = 256):
        super().__init__()
        self.img_list = glob(os.path.join(img_dir, "*.npy"))
        self.mask_list = glob(os.path.join(mask_dir, "*.npy"))
        self.exp = exp
        self.img_size = img_size

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        img = np.load(self.img_list[index]).astype(float).transpose(1, 2, 0)
        img = cv2.resize(img, (self.img_size, self.img_size)).transpose(2, 1, 0)
        img = eval_expression(self.exp, img.squeeze())
        img = TF.to_tensor(img)

        mask = np.load(self.mask_list[index]).astype(float)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = TF.to_tensor(mask)

        return img.float(), mask.float()


def jaccard_index(pred, target, threshold=0.5):
    pred_thresh = pred > threshold
    return torch.logical_and(pred_thresh, target).sum() / torch.logical_or(pred_thresh, target).sum()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_training_based_reward(exp: list, image_dir: str, mask_dir: str):
    unitless = check_unitless_validity(exp)
    if unitless is False:
        return -1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CustomDataset(image_dir, mask_dir, exp)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_set, val_set = torch.utils.data.random_split(dataset, [450, 63])

    train_loader = DataLoader(train_set, 8, True, worker_init_fn=seed_worker, num_workers=0)
    val_loader = DataLoader(val_set, 8, False, worker_init_fn=seed_worker, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
    model.segmentation_head[2].activation = torch.nn.Sigmoid()
    model = model.float()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    train_loss = val_loss = val_iou = 0
    for img, mask in tqdm(train_loader, "Training:"):
        img = img.to(device)
        mask = mask.to(device).squeeze()

        logits = model(img).squeeze()
        loss = criterion(logits, mask)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for img, mask in tqdm(val_loader, "Validation:"):
            img = img.to(device)
            mask = mask.to(device).squeeze()
            logits = model(img).squeeze()
            loss = criterion(logits, mask)
            val_loss += loss.item()
            val_iou += jaccard_index(logits, mask)

    print(
        f"Train loss: {train_loss / len(train_loader)}, Val loss: {val_loss / len(val_loader)}, Val IoU: {val_iou / len(val_loader)}"
    )

    return (val_iou / len(val_loader)).item()
