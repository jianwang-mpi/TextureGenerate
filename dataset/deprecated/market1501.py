# -*- coding:utf-8 -*-
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle
import nori2 as nori
from dataset.data_utils import ToTensor, RandomCrop, RandomFlip, Resize
from utils.imdecode import imdecode
from numpy.random import RandomState


# 读图

class Market1501Dataset(Dataset):
    
    def __init__(self, pkl_path = None, normalize=True,num_instance=4):
        
        self.normalize = normalize
        self.to_tensor = ToTensor(normalize=self.normalize)
        #self.data = []
        #self.generate_index()

        self.random_flip = RandomFlip(flip_prob=0.5)
        
        # 检查是否有该文件
        if not os.path.exists(pkl_path):
            raise ValueError('{} not exists!!'.format(pkl_path))
        # 打开pkl  pid:[_,image_id,camera_id]
        with open(pkl_path, 'rb') as fs:
            self.pkl = pickle.load(fs)
            
        self.sort_keys = list(sorted(self.pkl.keys()))
        
        self.len = len(self.pkl)
        
        # nori
        self.nf = nori.Fetcher()
        
        # 一次性一个人取多少张图片
        self.num_instance = num_instance


    def __getitem__(self, index):
        
        person_id = self.sort_keys[index] # 找到str的person id
        nori_ids_list = self.pkl[person_id]['nori_id']
        
        rng = RandomState()
        nori_ids = rng.choice(nori_ids_list, self.num_instance, replace=(len(nori_ids_list) < self.num_instance))
        
        
        img_list = []
        nori_list = []
        for nori_id in nori_ids:
        
            market_img = self.nf.get(nori_id)
            texture_img = imdecode(market_img)
            
            while texture_img is None or texture_img.shape[0] <= 0 or texture_img.shape[1] <= 0:
                
                new_nori_id = np.random.randint(0, len(nori_ids_list))
                market_img = self.nf.get(nori_ids[new_nori_id])
                texture_img = imdecode(market_img)
                
            texture_img = self.random_flip(texture_img)
            texture_img = self.to_tensor(texture_img)
            img_list.append(texture_img)
            nori_list.append(nori_id)
        
        idx_list = [index] * self.num_instance
        #texture_img_path = self.data[index]
        #texture_img = cv2.imread(texture_img_path)
        return img_list,idx_list

    def __len__(self):
        return self.len
        #return len(self.data)

    