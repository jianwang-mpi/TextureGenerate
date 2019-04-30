from __future__ import print_function, absolute_import

import glob
import re
from os import path as osp
import numpy as np
import pdb
import cv2
from torch.utils.data import Dataset
import pickle
import tqdm

class DeepFashion(object):
    
    dataset_dir = '/unsullied/sharefs/wangjian02/isilon-home/datasets/DeepFashion/DeepFashionSplit/'
    mask_dir = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/deepfashion-mask/'
    pose_dir = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/deepfashion-pose/'

    def __init__(self, root='data', **kwargs):
        
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d)')

        pid_container = set()
        for img_path in tqdm.tqdm(img_paths):

            # 对每一个 pattern.search(img_path).groups() 使用map函数
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            #assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            
            img_name = img_path[71:]
            img_name = img_name[img_name.find('/')+1:]
            mask_path = osp.join(self.mask_dir,img_name+'.npy')
            pose_path = osp.join(self.pose_dir,img_name+'.npy')
            
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, mask_path, pose_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


    