import numpy as np
import pickle
import os
import cv2
import torch
from smpl.render_texture import Renderer

action_files = [
    # '104/104_09.pkl',  # run
    '104/104_19.pkl',  # walk
    '39/39_14.pkl',  # walk
    # '36/36_32.pkl'     # up stairs
]
result = []
rotate_total_div = 8


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)))
    indices = indices.long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


for file_name in action_files:
    path = os.path.join('neutrSMPL_CMU', file_name)
    with open(path, 'rb') as f:
        data = pickle.load(f)
        renderer = Renderer('smpl/models/body.obj', 'smpl/models/neutral.pkl', w=224, h=224)
        texture_bgr = cv2.imread('smpl/models/default_texture2.jpg')
        texture_bgr = cv2.resize(texture_bgr, dsize=(224, 224))
        for rotate_div in range(0, rotate_total_div):
            for i in range(0, len(data['poses']), 20):
                thetas = np.concatenate((data['trans'][i], data['poses'][i], data['betas']))
                thetas[3:6] = [np.pi, 0, 0]
                rn, deviation, silhouette = renderer.render(thetas, texture_bgr,
                                                            rotate=np.array([0,
                                                                             2 * np.pi * rotate_div / rotate_total_div
                                                                                , 0]))

                result.append({
                    'mat': deviation,
                    'mask': silhouette
                })
            print('process: {} / {}'.format(rotate_div, rotate_total_div))

np.save('walk_224', result)
