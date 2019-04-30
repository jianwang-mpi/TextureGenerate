# import torch
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import random
from torch.autograd import Function
import os
import tqdm

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

    def forward(self, x, npy_paths,img_paths):
        # the input x is uv map batch of (N, C, H, W)
        # transfer it into (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # flat it and transpose it(H * W * C, N)
        
        x_flat = x.reshape(self.batch_size, -1).transpose(0, 1)
        
        
        
        result_flats = []
        masks = []
        
        for i in range(x_flat.shape[1]):
            
            
            #print(npy_paths[i])
            #print(img_paths[i])
            
            
            data = {}
            
            x_sing_flat = x_flat[:,i]
            x_sing_flat = x_sing_flat.unsqueeze(1)
            npy_path = npy_paths[i]
            action_npz_data = np.load(npy_path,encoding="latin1")
            
            
            action_npz_data.resize(1,)
            action_npz_data = action_npz_data[0]


        
            data['mat'] = self.sparse_mx_to_torch_sparse_tensor(action_npz_data['mat'])
            #data['bbox'] = self.bbox(action_npz_data['mask'][:, :, 0])
            data['mask'] = torch.from_numpy(action_npz_data['mask']).float().unsqueeze(0).permute(0, 3, 1, 2)

            if self.use_gpu:
                data['mat'] = data['mat'].cuda()
                data['mask'] = data['mask'].cuda()
                
            action_tensor = data
        
        
        
            mat = action_tensor['mat']
            mask = action_tensor['mask']
            #bbox = action_tensor['bbox']
            
            
            mat = nn.Parameter(mat, requires_grad=False)
            
            
            
            result_flat = DifferentialTextureRenderer.apply(x_sing_flat, mat)
            result_flat = result_flat.transpose(0, 1)
            
            masks.append(mask)
            result_flats.append(result_flat)
            
        masks = torch.cat(masks,dim=0)
        result_flats = torch.cat(result_flats,dim=0)
        
        # get the result of (NHWC)
        result = result_flats.reshape(self.batch_size, 128, 64, -1)
        # to NCHW
        result = result.permute(0, 3, 1, 2)
        
        return result, masks
    
    
    # train,isRandom is True  , test , isRandom is False

    def __init__(self, batch_size, use_gpu=False, bbox_size=(128, 64), center_random_margin=2):
        super(TextureToImage, self).__init__()
        print('start init the texture to image module')
        
        self.center_random_margin = center_random_margin
        
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        self.bbox_size = bbox_size
        
        
        
        

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
    uv_map_path = '/home/zhongyunshan/TextureGAN/TextureGAN/example_result'
    out_path = '/home/zhongyunshan/TextureGAN/TextureGAN/example_result_after'
    
    
    background = cv2.imread('/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/example_data/background.png')
    background = cv2.resize(background, (64, 128))
    
    
    tex_2_img = TextureToImage(batch_size=1,use_gpu=False)
    
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for root, dir, names in os.walk(uv_map_path):
        for name in names:
            full_path = os.path.join(root, name)
            

            texture_img = cv2.imread(full_path)
            texture_img = cv2.resize(texture_img, (64, 64))
            texture_img = torch.from_numpy(texture_img).unsqueeze(0).float()
            texture_img = texture_img.permute(0, 3, 1, 2)
            texture_img.requires_grad = True
            img, mask = tex_2_img(texture_img,['/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-pose/query/1448_c3s3_057278_00.jpg.npy'])
            
            
            img = img.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8)
            mask = mask.squeeze(0).permute(1, 2, 0).detach().numpy()
            img = img.astype(np.uint8)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            img = img * mask + background * (1 - mask)
            
            
            print(os.path.join(out_path, name))
            cv2.imwrite(os.path.join(out_path, name), img)

