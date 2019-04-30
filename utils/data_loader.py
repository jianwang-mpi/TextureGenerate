from __future__ import print_function, absolute_import

from dataset.data_utils import ToTensor, Resize

import cv2
from torch.utils.data import Dataset
import os
import numpy as np


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            #  do not change rgb for now!
            img = cv2.imread(img_path)
            
            #print(img_path)
            
            
            img = cv2.resize(img,(64,128))
            
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_deepfashion_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            
            
            img = cv2.imread(img_path)
            
            # for deepfashion dataset
            img = np.array(img)

            img = img[:,40:-40,:]

            img = cv2.resize(img,(64,128))
            
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_mask(mask_path):
    got_mask = False
    while not got_mask:
        try:
            mask = np.load(mask_path)
            
            # for deepfashion dataset
            
            mask = mask.astype(np.float)
            mask = cv2.resize(mask,(64,128),cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=2)
            mask = np.c_[mask, mask,mask]
            
            got_mask = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(mask_path))
            pass
    return mask


class ImageData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        
        self.normalize = True
        self.to_tensor = ToTensor(normalize=self.normalize)
        #self.random_flip = RandomFlip(flip_prob=0.5)
        

    def __getitem__(self, item):
        img_path, pose_path, pid, camid = self.dataset[item]
        
        # print(img_path, pose_path, pid, camid)
        
        img = read_image(img_path)
        
        
        #img = self.random_flip(img)
        img = self.to_tensor(img)
        
        
        
        
        return img,pose_path, pid, camid, img_path

    def __len__(self):
        return len(self.dataset)

class ImageData_deepfashoin_addmask(Dataset):
    def __init__(self, dataset, transform=None):
        
        
        self.normalize = True
        self.to_tensor = ToTensor(normalize=self.normalize)
        
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img_path, mask_path, pose_path, pid, camid  = self.dataset[item]
        
        if os.path.isdir(img_path):
            print('img_path',img_path)
            sys.exit(0)
        if os.path.isdir(mask_path):
            print('mask_path',mask_path)
            sys.exit(0)
       
        
        img = read_deepfashion_image(img_path)
        mask = read_mask(mask_path)
        
        
        
        if self.transform is not None:
            img,mask = self.transform(img,mask)
            
        img = self.to_tensor(img)
        
        mask = mask.transpose((2, 0, 1))
        
        return img,mask, pose_path, pid, camid, img_path, mask_path

    def __len__(self):
        return len(self.dataset)
