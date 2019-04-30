# import torch
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import random
from torch.autograd import Function
import os
from tqdm import tqdm
import sys


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

    # training time: isRandom is True; testing time: isRandom is False

    def __init__(self, action_npz, batch_size, img_size=224, use_gpu=False, bbox_size=(128, 64),
                 center_random_margin=2, isRandom=True):
        super(TextureToImage, self).__init__()
        print('start init the texture to image module')
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
        print('finish init the texture to image module')

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


if __name__ == '__main__':
    uv_map_path = sys.argv[1]
    out_path = sys.argv[2]

    background = cv2.imread('/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/example_data/background.png')
    background = cv2.resize(background, (224, 224))
    tex_2_img = TextureToImage(
        action_npz='/unsullied/sharefs/wangjian02/isilon-home/datasets/texture/tex_gan/walkfront_64.npy',
        batch_size=1,
        center_random_margin=2,
        isRandom=False)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for root, dir, names in os.walk(uv_map_path):
        for name in names:
            full_path = os.path.join(root, name)
            print(full_path)

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
