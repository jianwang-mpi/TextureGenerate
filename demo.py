import torch
import cv2
import argparse
import numpy as np
import os


class Demo:
    def __init__(self, model_path):
        print(model_path)

        self.model = torch.load(model_path).cuda()
        self.model.eval()

    def generate_texture(self, img_path):
        img = cv2.imread(img_path)

        print(img.shape)

        img = cv2.resize(img, (64, 128))
        img = (img / 225. - 0.5) * 2.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

        out = self.model(img)

        out = out.cpu().detach().numpy()[0]
        out = out.transpose((1, 2, 0))
        out = (out / 2.0 + 0.5) * 255.
        out = out.astype(np.uint8)
        out = cv2.resize(out, dsize=(64, 64))

        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show generated image')
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--img', '-i')
    parser.add_argument('--model', '-m', default='model_path')
    parser.add_argument('--out', '-o', default=None)

    args = parser.parse_args()
    img_path = args.img
    out_path = args.out
    model_path = args.model

    demo = Demo(model_path)

    print(img_path)
    if os.path.isdir(img_path):
        for root, dir, names in os.walk(img_path):
            for name in names:
                full_path = os.path.join(img_path, name)
                print(full_path)
                out = demo.generate_texture(img_path=full_path)

                print('out path', os.path.join(out_path, name))
                cv2.imwrite(os.path.join(out_path, name), out)
    else:
        out = demo.generate_texture(img_path=img_path)

        cv2.imshow('out', out)
        cv2.waitKey(0)
