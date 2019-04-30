import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset.data_utils import ToTensor, RandomCrop, RandomFlip, Resize

# 读图和读文件名（包含id）

class Market1501Dataset(Dataset):

    def __getitem__(self, index):
        texture_img_path = self.data[index]
        texture_img = cv2.imread(texture_img_path)
        if texture_img is None or texture_img.shape[0] <= 0 or texture_img.shape[1] <= 0:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        texture_img = self.random_flip(texture_img)
        texture_img = self.to_tensor(texture_img)
        return texture_img_path, texture_img

    def __len__(self):
        return len(self.data)

    def __init__(self, data_path_list, normalize=True):
        self.data_path_list = data_path_list
        self.normalize = normalize
        self.to_tensor = ToTensor(normalize=self.normalize)
        self.data = []
        self.generate_index()

        self.random_flip = RandomFlip(flip_prob=0.5)

    def generate_index(self):
        print('generating market 1501 index')
        for data_path in self.data_path_list:
            for root, dirs, files in os.walk(data_path):
                for name in files:
                    if name.endswith('.jpg'):
                        self.data.append(os.path.join(root, name))

        print('finish generating market 1501 index, found texture image: {}'.format(len(self.data)))


