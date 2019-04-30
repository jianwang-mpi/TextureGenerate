import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from .data_utils import ToTensor

import tqdm


class RealTextureDataset(Dataset):


    def __getitem__(self, index):
        texture_img_path = self.data[index]
        texture_img = cv2.imread(texture_img_path)
        texture_img = cv2.resize(texture_img, dsize=(self.img_size, self.img_size))

        texture_img = self.to_tensor(texture_img)

        return texture_img

    def __len__(self):
        return len(self.data)

    def __init__(self, data_path, img_size=64, normalize=True):
        self.data_path = data_path
        self.img_size = img_size
        self.normalize = normalize
        self.to_tensor = ToTensor(normalize=self.normalize)
        self.data = []
        self.generate_index()

    def generate_index(self):
        print('generating index')
        for root, dirs, files in os.walk(self.data_path):
            for name in tqdm.tqdm(files):
                if name.endswith('.jpg') and 'nongrey' in name:
                    self.data.append(os.path.join(root, name))

        print('finish generating index, found texture image: {}'.format(len(self.data)))



# -*- coding:utf-8 -*-
#
#
# import os
#
# import cv2
# import numpy as np
# from torch.utils.data import Dataset
# import pickle
# import nori2 as nori
# from utils.imdecode import imdecode
# from .data_utils import ToTensor
#
#
# # 真实的uvmap
#
# class RealTextureDataset(Dataset):
#
#     def __init__(self, data_path=None, img_size=64, pkl_path=None, normalize=True):
#         # self.data_path = data_path
#         self.img_size = img_size
#         self.normalize = normalize
#
#         self.to_tensor = ToTensor(normalize=self.normalize)
#
#         # 检查是否有该文件
#         if not os.path.exists(pkl_path):
#             raise ValueError('{} not exists!!'.format(pkl_path))
#         # 打开pkl  pid:[_,image_id,camera_id]
#         with open(pkl_path, 'rb') as fs:
#             self.pkl = pickle.load(fs)
#         self.len = len(self.pkl)
#
#         # nori
#         self.nf = nori.Fetcher()
#
#     def __getitem__(self, index):
#         texture_img = self.nf.get(self.pkl[index][0])
#
#         # decode
#         texture_img = imdecode(texture_img)
#         texture_img = cv2.resize(texture_img, dsize=(self.img_size, self.img_size))
#
#         texture_img = self.to_tensor(texture_img)
#
#         return texture_img
#
#     def __len__(self):
#         return self.len
