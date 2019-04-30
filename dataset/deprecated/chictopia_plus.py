import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset.data_utils import ToTensor, Resize


class ChictopiaPlusDataset(Dataset):

    def bbox(self, mask):
        rows = np.any(mask, axis=0)
        cols = np.any(mask, axis=1)
        cmin, cmax = np.where(rows)[0][[0, -1]]
        rmin, rmax = np.where(cols)[0][[0, -1]]

        h = rmax - rmin
        w = int(h / 2)

        r_center = float(rmax + rmin) / 2
        c_center = float(cmax + cmin) / 2

        rmin = int(r_center - h / 2)
        rmax = int(r_center + h / 2)

        cmin = int(c_center - w / 2)
        cmax = int(c_center + w / 2)

        return (cmin, rmin), (cmax, rmax)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = cv2.imread(img_path)
        segment_img_path = img_path.replace('image:png', 'bodysegments')
        segment_img = cv2.imread(segment_img_path)
        mask = (segment_img >= 1).astype(np.float)

        tl, br = self.bbox(mask)
        img = img[tl[1]: br[1], tl[0]:br[0], :]
        mask = mask[tl[1]: br[1], tl[0]: br[0], :]

        if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
            return self.__getitem__(np.random.randint(0, self.__len__()))

        img = self.resize(img)
        mask = self.resize(mask)

        img = self.to_tensor(img)
        mask = self.mask_to_tensor(mask)

        return img, mask

    def __len__(self):
        return len(self.data)

    def __init__(self, data_path, img_size=(128, 64), normalize=True):
        self.data_path = data_path
        self.img_size = img_size
        self.normalize = normalize
        self.resize = Resize(self.img_size)
        self.to_tensor = ToTensor(normalize=self.normalize)
        self.mask_to_tensor = ToTensor(normalize=False)
        self.data = []
        self.generate_index()

    def generate_index(self):
        print('generating ChictopiaPlus index')
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                if name.endswith('.png') and 'image' in name:
                    self.data.append(os.path.join(root, name))

        print('finish generating index, found texture image: {}'.format(len(self.data)))


if __name__ == '__main__':
    dataset = ChictopiaPlusDataset('/unsullied/sharefs/wangjian02/isilon-home/datasets/ChictopiaPlus/train')
    img, mask = dataset.__getitem__(1)
    img = img.permute(1, 2, 0).numpy()
    mask = mask.permute(1, 2, 0).numpy()

    img = img / 2.0 + 0.5
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.imshow('mask', mask)
    cv2.waitKey()
