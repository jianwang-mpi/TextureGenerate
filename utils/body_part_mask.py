import os
import torch
import cv2
import numpy as np


class TextureMask:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.part = {
            'face': 'models/face_mask.png',
            'hand': 'models/hand_mask.png',
            'body': 'models/body_mask.png',
            'short_up': 'models/short_up_mask.jpg',
            'short_trouser': 'models/short_trouser_mask.jpg'
        }

    def get_mask(self, part):
        mask_path = self.part[part]
        mask = cv2.imread(mask_path)

        mask = cv2.resize(mask, self.size)
        mask = mask / 255.
        mask = mask.transpose((2, 0, 1))
        mask = np.expand_dims(mask, 0)
        mask = torch.from_numpy(mask).float()
        return mask

    def get_numpy_mask(self, part):
        mask_path = self.part[part]
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, self.size)
        mask = mask / 255.
        return mask


if __name__ == '__main__':
    masker = TextureMask(size=64)
    mask = masker.get_mask("face")
    cv2.imshow('mask', mask)
    cv2.waitKey()
