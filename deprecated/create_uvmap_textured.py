import torch
import cv2
import argparse

import numpy as np
import os
import sys
import tqdm
import os

import torch.nn as nn
import time
import random
from torch.autograd import Function
import os
from dataset.market1501_pose_split_test import Market1501Dataset


class DifferentialTextureRenderer(Function):

    @staticmethod
    def forward(ctx, texture_img_flat, render_sparse_matrix):
        result = torch.mm(render_sparse_matrix, texture_img_flat)
        ctx.save_for_backward(render_sparse_matrix)
        return result

    @staticmethod
    def backward(ctx, grad_outputs):
        render_sparse_matrix = ctx.saved_tensors[0]
        result = torch.mm(render_sparse_matrix.transpose(0, 1), grad_outputs)
        return result, None


class TextureToImage(nn.Module):

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)))
        indices = indices.long()
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, x):
        # the input x is uv map batch of (N, C, H, W)
        # transfer it into (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # flat it and transpose it(H * W * C, N)
        x_flat = x.reshape(self.batch_size, -1).transpose(0, 1)
        if self.isRandom:
            action_tensor = random.choice(self.action_sparse_tensor_data)
        else:
            action_tensor = self.action_sparse_tensor_data[0]
        mat = action_tensor['mat']
        mask = action_tensor['mask']
        bbox = action_tensor['bbox']
        mat = nn.Parameter(mat, requires_grad=False)
        result_flat = DifferentialTextureRenderer.apply(x_flat, mat)
        result_flat = result_flat.transpose(0, 1)
        # get the result of (NHWC)
        result = result_flat.reshape(self.batch_size, self.img_size, self.img_size, -1)
        # to NCHW
        result = result.permute(0, 3, 1, 2)
        return result, mask, bbox

    # train,isRandom is True  , test , isRandom is False

    def __init__(self, action_npz, batch_size, img_size=224, use_gpu=False, bbox_size=(128, 64),
                 center_random_margin=2, isRandom=True):
        super(TextureToImage, self).__init__()
        # print('start init the texture to image module')
        action_npz_data = np.load(action_npz, encoding="latin1")
        self.center_random_margin = center_random_margin
        self.action_sparse_tensor_data = []
        self.batch_size = batch_size
        self.img_size = img_size
        self.bbox_size = bbox_size
        self.isRandom = isRandom

        for data in action_npz_data:
            data['mat'] = self.sparse_mx_to_torch_sparse_tensor(data['mat'])
            data['bbox'] = self.bbox(data['mask'][:, :, 0])
            data['mask'] = torch.from_numpy(data['mask']).float() \
                .unsqueeze(0).permute(0, 3, 1, 2).repeat(self.batch_size, 1, 1, 1)

            if use_gpu:
                data['mat'] = data['mat'].cuda()
                data['mask'] = data['mask'].cuda()
            self.action_sparse_tensor_data.append(data)
        # print('finish init the texture to image module')

    def bbox(self, img):
        h = self.bbox_size[0]
        w = self.bbox_size[1]
        rows = np.any(img, axis=0)
        cols = np.any(img, axis=1)
        cmin, cmax = np.where(rows)[0][[0, -1]]
        rmin, rmax = np.where(cols)[0][[0, -1]]

        r_center = float(rmax + rmin) / 2 + random.randint(-self.center_random_margin, 0)
        c_center = float(cmax + cmin) / 2 + random.randint(0, self.center_random_margin)

        rmin = int(r_center - h / 2)
        rmax = int(r_center + h / 2)

        cmin = int(c_center - w / 2)
        cmax = int(c_center + w / 2)

        return (cmin, rmin), (cmax, rmax)

    def test(self):
        texture_img = cv2.imread('models/default_texture2.jpg')
        texture_img = torch.from_numpy(texture_img).unsqueeze(0).float()
        texture_img = texture_img.reshape(1, -1).transpose(0, 1)
        start_time = time.time()

        action_tensor = random.choice(self.action_sparse_tensor_data)['mat']
        result_flat = torch.smm(action_tensor, texture_img).to_dense()
        result_flat = result_flat.transpose(0, 1)
        result_flat = result_flat.reshape(1, 224, 224, 3)
        stop_time = time.time()
        print('time use: {}'.format(stop_time - start_time))
        result_flat = result_flat.numpy()[0, :]
        cv2.imshow('result', result_flat.astype(np.uint8))
        cv2.waitKey()


class Demo:
    def __init__(self, model_path, z_size=1024):
        print(model_path)

        self.model = torch.load(model_path)

        self.model.eval()
        self.z_size = z_size

    def generate_texture(self, img_path):
        img = cv2.imread(img_path)

        if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
            return 0, 0

        img = cv2.resize(img, (64, 128))
        img = (img / 225. - 0.5) * 2.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

        out = self.model(img)
        out = out.cpu().detach().numpy()[0]
        out = out.transpose((1, 2, 0))
        out = (out / 2.0 + 0.5) * 255.
        out = out.astype(np.uint8)
        out = cv2.resize(out, dsize=(64, 64))

        return out, 1


def create_dir(uvmap_dir, textured_dir):
    if not os.path.exists(uvmap_dir):
        os.mkdir(uvmap_dir)

    if not os.path.exists(textured_dir):
        os.mkdir(textured_dir)


def read_background():
    data_path = '/unsullied/sharefs/wangjian02/isilon-home/datasets/SURREAL/smpl_data/textures'

    PRW_img_path = '/unsullied/sharefs/wangjian02/isilon-home/datasets/PRW/frames'
    CUHK_SYSU_path = '/unsullied/sharefs/wangjian02/isilon-home/datasets/CUHK-SYSU'

    data_path_list = [PRW_img_path, CUHK_SYSU_path]

    backgrounds = []

    for data_path in data_path_list:
        for root, dirs, files in os.walk(data_path):
            for name in files:
                if name.endswith('.jpg'):
                    backgrounds.append(os.path.join(root, name))

    return backgrounds


def create_uvmap(model_path, uvmap_dir):
    demo = Demo(model_path)

    dataset = Market1501Dataset()
    input_imgs = dataset.train

    out_path = uvmap_dir

    print('len of input images', len(input_imgs))

    for full_path in tqdm.tqdm(input_imgs):

        p = full_path[0]
        out, flag = demo.generate_texture(img_path=p)
        if flag == 0:
            continue

        name = p[p.find('/', 68) + 1:]
        cv2.imwrite(os.path.join(out_path, name), out)


def create_textured(uvmap_dir, textured_dir, backgrounds):
    uv_map_path = uvmap_dir
    out_path = textured_dir

    tex_2_img = TextureToImage(
        action_npz='/unsullied/sharefs/wangjian02/isilon-home/datasets/texture/tex_gan/walk_64.npy',
        batch_size=1,
        center_random_margin=2,
        isRandom=False)

    count = 0
    for root, dir, names in os.walk(uv_map_path):
        for name in tqdm.tqdm(names):
            background = cv2.imread(backgrounds[np.random.randint(len(backgrounds), size=1)[0]])
            background = cv2.resize(background, (224, 224))

            '''
            background[:,:,0] = 255
            background[:,:,1] = 255
            background[:,:,2] = 255
            '''

            count += 1
            full_path = os.path.join(root, name)

            texture_img = cv2.imread(full_path)
            texture_img = cv2.resize(texture_img, (64, 64))
            texture_img = torch.from_numpy(texture_img).unsqueeze(0).float()
            texture_img = texture_img.permute(0, 3, 1, 2)
            texture_img.requires_grad = True
            img, mask, bbox = tex_2_img(texture_img)

            img = img.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8)
            mask = mask.squeeze(0).permute(1, 2, 0).detach().numpy()
            c_center = (bbox[0][0] + bbox[1][0]) / 2
            r_center = (bbox[0][1] + bbox[1][1]) / 2

            img = img.astype(np.uint8)

            img = img * mask + background * (1 - mask)
            tl, br = bbox
            img = img[tl[1]:br[1], tl[0]:br[0], :]
            cv2.imwrite(os.path.join(out_path, name), img)


def run():
    '''
    model_names = ['ImageNet_PerLoss2018-10-23_18:14:53.982469/2018-10-24_10:30:39.835040_epoch_120',
                   'NoPCB_PerLoss2018-10-23_18:16:04.651977/2018-10-24_06:20:33.259434_epoch_120',
                   'PCB_2048_256_L12018-10-23_18:13:29.746996/2018-10-24_05:17:39.706192_epoch_120',
                   'PCB_ALLCat_PerLoss2018-10-23_18:17:51.451793/2018-10-24_09:42:22.511739_epoch_120',
                   'PCB_PerLoss2018-10-23_18:16:59.216650/2018-10-24_13:27:16.867817_epoch_120',
                   'PCB_PerLoss_NoPosed2018-10-24_11:01:36.682130/2018-10-24_12:27:34.799378_epoch_120',
                   'PCB_RGB_L12018-10-23_18:12:42.827038/2018-10-23_23:51:33.516745_epoch_120',
                   'PCB_softmax2018-10-23_18:18:39.775789/2018-10-24_05:05:52.977378_epoch_120',
                   'PCB_TripletHard2018-10-23_18:20:48.070572/2018-10-24_04:35:05.054042_epoch_120']
    '''

    model_names = ['PCB_256_L12018-11-16_17:53:20.894085/2018-11-17_05:16:20.990883_epoch_120']

    model_root = '/unsullied/sharefs/zhongyunshan/isilon-home/model-parameters/Texture'

    for model_name in model_names:
        model_path = os.path.join(model_root, model_name)

        model = model_path[model_path.find('/', 61) + 1:model_path.find('/', 69)] + '_' + model_path[
                                                                                          model_path.find('epoch'):]
        # model = model+'_all'
        uvmap_root = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-uvmap'
        textured_root = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-textured'

        uvmap_dir = os.path.join(uvmap_root, model)
        textured_dir = os.path.join(textured_root, model)

        print('model', model_name)
        print('uvmap_root', uvmap_root)
        print('textured_root', textured_root)

        print('uvmap_dir', uvmap_dir)
        print('textured_dir', textured_dir)

        print('create dir')
        create_dir(uvmap_dir, textured_dir)

        print('create uvmap')
        create_uvmap(model_path, uvmap_dir)

        print('read backgrounds')
        backgrounds = read_background()

        print('create textued img')
        create_textured(uvmap_dir, textured_dir, backgrounds)


run()
