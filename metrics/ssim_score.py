from PIL import Image
from torch.utils.data import Dataset
import glob
import re
from os import path as osp
import numpy as np
import pdb
import os
import cv2

from pytorch_ssim_master import pytorch_ssim
import torch
from torch.autograd import Variable
import tqdm
from multiprocessing import Pool


def process_dir(dir_path, relabel=False):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths:

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
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel: pid = pid2label[pid]
        dataset.append((img_path, pid, camid))

    num_pids = len(pid_container)
    num_imgs = len(dataset)
    return dataset, num_pids, num_imgs


def get_data(dataset_dir):
    train, num_train_pids, num_train_imgs = process_dir(dataset_dir, relabel=True)

    num_total_pids = num_train_pids
    num_total_imgs = num_train_imgs

    print("=> Market1501 loaded")
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))

    print("  ------------------------------")
    print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
    print("  ------------------------------")

    return train


def fun(root, model, ori_train):
    print(model)

    scores = []

    dataset = ori_train

    path = os.path.join(root, model)

    for item in tqdm.tqdm(dataset):
        img_ori = cv2.imread(item[0])
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        img_ori = torch.from_numpy(np.rollaxis(img_ori, 2)).float().unsqueeze(0) / 255.0

        p = item[0]
        p = p[p.find('market-origin-ssim'):]
        p = p[p.find('/') + 1:]

        p = os.path.join(path, p)

        img_oth = cv2.imread(p)
        img_oth = cv2.cvtColor(img_oth, cv2.COLOR_BGR2RGB)
        img_oth = torch.from_numpy(np.rollaxis(img_oth, 2)).float().unsqueeze(0) / 255.0

        ssim_loss = pytorch_ssim.SSIM(window_size=11)

        scores.append(ssim_loss(img_ori, img_oth))

    return model, np.mean(scores)


root = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-textured-ssim'

ori_train = get_data('/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-origin-ssim')

results = []

# model = 'no_face2018-11-09_10:57:53.148362_epoch_120'
# result = fun(root,model,ori_train)
# results.append(result)
for model in os.listdir(root):

    if model != 'PCB_256_L12018-11-16_17:53:20.894085_epoch_120':
        continue

    result = fun(root, model, ori_train)
    results.append(result)

for i in results:
    print(i)
