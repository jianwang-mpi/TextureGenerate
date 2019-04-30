import torch.nn as nn
import torch

# 无用

class ClothesColorVarLoss(nn.Module):
    def forward(self, image):
        total_var = 0
        for i, item in enumerate(image):
            for j, channel in enumerate(item):
                up_channel = channel[self.short_up_mask[i, j]]
                trouser_channel = channel[self.short_trouser_mask[i, j]]
                total_var += torch.var(up_channel) + torch.var(trouser_channel)
        return total_var / (2 * image.shape[0] * image.shape[1])

    def __init__(self, texture_mask, use_gpu):
        super(ClothesColorVarLoss, self).__init__()
        self.short_up_mask = texture_mask.get_mask('short_up')
        self.short_trouser_mask = texture_mask.get_mask('short_trouser')

        self.short_up_mask = self.short_up_mask.type(torch.ByteTensor)
        self.short_trouser_mask = self.short_trouser_mask.type(torch.ByteTensor)
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.short_up_mask = self.short_up_mask.cuda()
            self.short_trouser_mask = self.short_trouser_mask.cuda()

if __name__ == '__main__':
    from utils.body_part_mask import TextureMask
    texture_mask = TextureMask(size=64, batch_size=4)
    color_loss = ClothesColorVarLoss(texture_mask, False)
    img = torch.randn(4, 3, 64, 64).float()
    result = color_loss(img)
    print(result)
