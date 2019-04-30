# -*- coding:utf-8 -*-
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from dataset.data_utils import ToTensor, RandomCrop, RandomFlip, Resize
import pickle
import nori2 as nori
from utils.imdecode import imdecode
from numpy.random import RandomState

# 读图，把图中的人的bounding box截掉，返回

class PRWDataset(Dataset):
    
    def __init__(self,img_size=(128, 64), bbox_threshold=200, pkl_path = None,normalize=True,num_instance=4):
        
        self.img_size = img_size
        self.normalize = normalize
        self.to_tensor = ToTensor(normalize=self.normalize)
        
        self.bbox_threshold = bbox_threshold
        

        self.random_flip = RandomFlip(flip_prob=0.5)
        self.resize = Resize(output_size=self.img_size)
        
        # 检查是否有该文件
        if not os.path.exists(pkl_path):
            raise ValueError('{} not exists!!'.format(pkl_path))
        # 打开pkl  pid:[_,image_id,camera_id]
        with open(pkl_path, 'rb') as fs:
            self.pkl = pickle.load(fs)
            
            
        
        self.len = len(self.pkl)
        
        # nori
        self.nf = nori.Fetcher()
        
        # 一次性一个人取多少张图片
        self.num_instance = num_instance

    def isReChoice(self,img,bbox):
        
        while img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
            
            return True
        
        x = int(bbox[1])
        y = int(bbox[2])
        w = int(bbox[3])
        h = int(bbox[4])
        img = img[y:y + h, x:x + w]
            
        while img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
            return True
        
        return False
        
        

    def __getitem__(self, index):
        
        
        
        
        items_list = self.pkl[index]
        
        rng = RandomState()
        items_ids = rng.choice(len(items_list), self.num_instance, replace=(len(items_list) < self.num_instance))
        
        
        img_list = []
        nori_list = []
        for items_id in items_ids:
            
            
            raw = self.nf.get(items_list[items_id][0])
            
            
            
            img,bbox = pickle.loads(raw)
            #img = imdecode(img)
            
            while self.isReChoice(img,bbox):
                
                
                # re select
                new_items_id = np.random.randint(0, len(items_list))
                
                raw = self.nf.get(items_list[new_items_id][0])
                img,bbox = pickle.loads(raw)
                #img = imdecode(img)
                #img = img[:, :, ::-1]  # BGR to RGBs
            
            # 裁剪
            x = int(bbox[1])
            y = int(bbox[2])
            w = int(bbox[3])
            h = int(bbox[4])
            img = img[y:y + h, x:x + w]

            img = self.resize(img)
            # img = self.random_flip(img) 原本就没有加
            img = self.to_tensor(img)
            
            img_list.append(img)
            nori_list.append(items_list[items_id][0])
            
        idx_list = [index] * self.num_instance
        return img_list,idx_list

    def __len__(self):
        return self.len

    '''
    def generate_index(self):
        print('generating prw index')

        for root, dirs, files in os.walk(self.frames_path):
            for name in files:
                if name.endswith('.jpg'):
                    img_path = os.path.join(root, name)
                    anno_name = name + '.mat'
                    anno_path = os.path.join(self.annotation_path, anno_name)
                    anno_mat = loadmat(anno_path)
                    if 'box_new' in anno_mat:
                        bboxs = anno_mat['box_new']
                    elif 'anno_file' in anno_mat:
                        bboxs = anno_mat['anno_file']
                    else:
                        continue
                    for bbox in bboxs:
                        self.data.append({'img_path': img_path,
                                          'bbox': bbox
                                          })

        print('finish generating PRW index, found texture image: {}'.format(len(self.data)))
    '''

if __name__ == '__main__':
    dataset = PRWDataset('/unsullied/sharefs/wangjian02/isilon-home/datasets/PRW')
    for i in range(10):
        img = dataset.__getitem__(i * 300)
        img = img.permute(1, 2, 0).detach().numpy()
        img = img / 2.0 + 0.5
        cv2.imshow('img', img)
        cv2.waitKey(0)
