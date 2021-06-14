from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np

from PIL import Image
import os
from glob import glob
import random

day_ims = os.path.join("/content/drive/MyDrive/Night2Day/day", "day")
night_ims = os.path.join("/content/drive/MyDrive/Night2Day/night", "night")


def get_transforms():
    tfms = [
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        T.Resize((168, 168)),
        T.ColorJitter(0.1, 0.1, 0.1)
    ]
    return tfms


class ImageDataset(Dataset):
    def __init__(self, tfms=None, unaligned=True, mode='train'):
        super().__init__()
        if tfms is None:
            self.transform = T.Compose(get_transforms())
        else:
            assert type(tfms) == list, "Please send a list of transforms"
            self.transform = T.Compose(*tfms)
        self.unaligned = unaligned
        self.files_A = sorted(glob(os.path.join(night_ims, mode) + "/*.*"))
        self.files_B = sorted(glob(os.path.join(day_ims, mode) + "/*.*"))
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def __getitem__(self, index):
        A = np.array(Image.open(self.files_A[index % self.len_A]))
        if self.unaligned:
            B = np.array(Image.open(self.files_B[random.randint(0, self.len_B-1)]))
        else:
            B = np.array(Image.open(self.files_B[index % self.len_B]))
        item_A = self.transform(A)
        item_B = self.transform(B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(self.len_A, self.len_B)
