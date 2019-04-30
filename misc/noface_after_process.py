import os
from utils.body_part_mask import TextureMask
import cv2
import numpy as np

texture_mask = TextureMask(size=64)
face_mask = texture_mask.get_numpy_mask('face')
hand_mask = texture_mask.get_numpy_mask('hand')
mask = face_mask + hand_mask

uv_map_path = '/home/wangjian02/Projects/TextureGAN/tmp/test_img/uv_no_face'
out_path = '/home/wangjian02/Projects/TextureGAN/tmp/test_img/uv_no_face_process'

gt_path = '/home/wangjian02/Projects/TextureGAN/models/nongrey_male_0002.jpg'
gt_img = cv2.imread(gt_path)
gt_img = cv2.resize(gt_img, dsize=(64, 64))

if not os.path.exists(out_path):
    os.mkdir(out_path)

for root, dir, names in os.walk(uv_map_path):
    for name in names:
        full_path = os.path.join(root, name)
        print(full_path)

        texture_img = cv2.imread(full_path)
        texture_img = cv2.resize(texture_img, (64, 64))

        new_img = texture_img * (1 - mask) + gt_img * mask
        new_img = new_img.astype(np.uint8)

        cv2.imwrite(os.path.join(out_path, name), new_img)
