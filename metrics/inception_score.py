import glob
import os
import pickle
import re
from os import path as osp

import numpy as np
import torch
import tqdm
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3

from dataset.market1501_pose_split_train import Market1501Dataset
from utils.data_loader import ImageData


class Market1501Dataset(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    pose_dataset_dir = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-pose/'

    pkl_path = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/saveForTest.pkl'

    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir

        print(self.pkl_path)

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.dataset_dir, relabel=True,
                                                                  pkl_path=self.pkl_path)

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  ------------------------------")

        self.train = train

        self.num_train_pids = num_train_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_dir(self, dir_path, relabel=False, pkl_path=None):

        if pkl_path is not None:
            with open(pkl_path, 'rb') as f:
                saveForTest = pickle.load(f)
        else:
            saveForTest = []

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:

            # 对每一个 pattern.search(img_path).groups() 使用map函数
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1 or pid not in saveForTest:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:

            img_name = img_path[67:]
            img_name = img_name[img_name.find('/') + 1:]

            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 or pid not in saveForTest:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, '', pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


def inception_score(cuda=True, batch_size=128, resize=True, splits=5):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    assert batch_size > 0

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    temp = []

    # root = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-uvmap/'
    # root = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-textured-ssim'
    root = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-textured-ssim'

    for d in os.listdir(root):

        print('model', d)

        if d != 'PCB_256_L12018-11-16_17:53:20.894085_epoch_120':
            continue

        p = os.path.join(root, d)

        dataset = Market1501Dataset(p)  # test

        dataloader = DataLoader(
            ImageData(dataset.train),
            batch_size=32, num_workers=2,
            pin_memory=True
        )

        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval()
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

        def get_pred(x):
            if resize:
                x = up(x)
            x = inception_model(x)
            return F.softmax(x).data.cpu().numpy()

        preds = []

        for i, batch in tqdm.tqdm(enumerate(dataloader, 0)):
            imgs, pids, _, _, _ = batch
            imgs = imgs.cuda()

            preds.append(get_pred(imgs))

        preds = np.concatenate(preds)
        # Now compute the mean kl-div
        split_scores = []

        N = len(preds)

        print('len of preds', len(preds))

        for k in range(splits):

            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        temp.append((d, np.mean(split_scores), np.std(split_scores)))
    return temp


temp = inception_score(cuda=True, batch_size=128, resize=True, splits=10)

for i in temp:
    print(i)
