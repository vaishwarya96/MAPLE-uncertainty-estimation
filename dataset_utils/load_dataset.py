import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

from config import get_cfg_defaults 
cfg = get_cfg_defaults()

class LoadDataset(data.Dataset):
    def __init__(self, X, y, train=True):

        self.image_paths, self.class_ids = X, y
        self.train = train

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        class_id = self.class_ids[idx]
        img = Image.open(img_path).convert('RGB')
        if self.train:
            image_transform = self.train_img_transforms()
        else:
            image_transform = self.test_img_transforms()
        img = image_transform(img)

        return img, class_id, img_path

    def train_img_transforms(self):

        trans = []
        trans.append(transforms.Resize(size=cfg.DATASET.IMG_SIZE))
        trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=cfg.DATASET.IMG_MEAN, std=cfg.DATASET.IMG_STD))
        trans = transforms.Compose(trans)

        return trans

    def test_img_transforms(self):

        trans = []
        trans.append(transforms.Resize(size=cfg.DATASET.IMG_SIZE))
        trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=cfg.DATASET.IMG_MEAN, std=cfg.DATASET.IMG_STD))
        trans = transforms.Compose(trans)

        return trans