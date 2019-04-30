import numpy as np
import torch
from torch import nn
from torchvision.models import resnet, vgg


def conv3x3(in_planes, out_planes, stride=1, padding=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=has_bias)


def conv5x5(in_features, out_features, stride=1, padding=2, has_bias=False):
    # 5x5 convolution with padding
    return nn.Conv2d(in_features, out_features, kernel_size=5, stride=stride, padding=padding, bias=has_bias)


def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False):
    # 7x7 convolution with padding
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=7, stride=stride, padding=padding, bias=has_bias)


def conv_bn_relu(in_planes, out_planes, kernel_size=3, stride=1):
    if kernel_size == 3:
        conv = conv3x3(in_planes, out_planes, stride, 1)
    elif kernel_size == 5:
        conv = conv5x5(in_planes, out_planes, stride, 2)
    elif kernel_size == 7:
        conv = conv7x7(in_planes, out_planes, stride, 3)
    else:
        return None
    nn.init.xavier_uniform_(conv.weight)
    bn = nn.BatchNorm2d(out_planes)
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)
    relu = nn.LeakyReLU(inplace=True, negative_slope=0.02)
    return nn.Sequential(conv, bn, relu)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=3, scale_factor=2):
        super(EncoderBlock, self).__init__()
        if scale_factor == 2:
            self.block = nn.Sequential(
                conv_bn_relu(in_channels, middle_channels, kernel_size=kernel_size),
                conv_bn_relu(middle_channels, out_channels, kernel_size=kernel_size),
                nn.MaxPool2d(kernel_size=2)
            )
        elif scale_factor == 4:
            self.block = nn.Sequential(
                conv_bn_relu(in_channels, middle_channels, kernel_size=kernel_size),
                nn.MaxPool2d(kernel_size=2),
                conv_bn_relu(middle_channels, out_channels, kernel_size=kernel_size),
                nn.MaxPool2d(kernel_size=2)
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 kernel_size=3,
                 scale_factor=2,
                 upsample='upsample'):

        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        if upsample == 'deconv':
            if scale_factor == 2:
                self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, middle_channels, bias=False, kernel_size=2, stride=2),
                    nn.BatchNorm2d(middle_channels),
                    nn.LeakyReLU(inplace=True, negative_slope=0.02),
                    conv_bn_relu(middle_channels, out_channels, kernel_size=kernel_size),
                )
            elif scale_factor == 4:
                self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, middle_channels,
                                       kernel_size=kernel_size, bias=False),
                    nn.BatchNorm2d(middle_channels),
                    nn.LeakyReLU(inplace=True, negative_slope=0.02),
                    nn.ConvTranspose2d(in_channels, middle_channels,
                                       kernel_size=kernel_size, bias=False),
                    nn.BatchNorm2d(middle_channels),
                    nn.LeakyReLU(inplace=True, negative_slope=0.02),
                )
        else:
            if scale_factor == 2:
                self.block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    conv_bn_relu(in_channels, middle_channels, kernel_size=kernel_size),
                    conv_bn_relu(middle_channels, out_channels, kernel_size=kernel_size),
                )
            elif scale_factor == 4:
                self.block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    conv_bn_relu(in_channels, middle_channels, kernel_size=kernel_size),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    conv_bn_relu(middle_channels, out_channels, kernel_size=kernel_size),
                )

    def forward(self, x):
        return self.block(x)


class Generator_(torch.nn.Module):
    def __init__(self, input_dimensions, output_channels):
        super(Generator_, self).__init__()
        self.in_dimension = input_dimensions
        self.out_channels = output_channels
        self.linear = nn.Linear(input_dimensions, out_features=1024)
        self.relu = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.bn = nn.BatchNorm1d(num_features=1024)
        self.model = nn.Sequential(
            DecoderBlock(in_channels=1024, middle_channels=512, out_channels=512, kernel_size=5),
            DecoderBlock(in_channels=512, middle_channels=256, out_channels=256, kernel_size=5),
            DecoderBlock(in_channels=256, middle_channels=128, out_channels=128, kernel_size=5),
            DecoderBlock(in_channels=128, middle_channels=64, out_channels=64, kernel_size=5),
            DecoderBlock(in_channels=64, middle_channels=32, out_channels=32, kernel_size=5),
            DecoderBlock(in_channels=32, middle_channels=16, out_channels=8, kernel_size=5),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.bn(self.relu(self.linear(x)))
        x = x.view(-1, 1024, 1, 1)
        x = self.model(x)
        return x


class Discriminator_(torch.nn.Module):
    def __init__(self, input_channels, out_dimension):
        super(Discriminator_, self).__init__()
        self.input_channels = input_channels
        self.output_dimension = out_dimension
        self.feature = nn.Sequential(
            EncoderBlock(in_channels=3, middle_channels=8, out_channels=8, kernel_size=5),
            EncoderBlock(in_channels=8, middle_channels=16, out_channels=16, kernel_size=5),
            EncoderBlock(in_channels=16, middle_channels=32, out_channels=32, kernel_size=5),
            EncoderBlock(in_channels=32, middle_channels=64, out_channels=64, kernel_size=5),
            EncoderBlock(in_channels=64, middle_channels=128, out_channels=128, kernel_size=5),
            EncoderBlock(in_channels=128, middle_channels=256, out_channels=256, kernel_size=5)
        )

        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=512, out_features=out_dimension)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 512)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class Discriminator(nn.Module):
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

    def __init__(self, input_channels, output_dimension, gpu_ids=None):
        super(Discriminator, self).__init__()
        self.model = Discriminator_(input_channels, output_dimension)
        self.gpu_ids = gpu_ids


class Generator(nn.Module):
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

    def __init__(self, input_dimension, output_channels, gpu_ids=None):
        super(Generator, self).__init__()
        self.model = Generator_(input_dimension, output_channels)
        self.gpu_ids = gpu_ids


if __name__ == '__main__':
    generator = Generator(input_dimension=1024, output_channels=3)
    discriminator = Discriminator(input_channels=3, output_dimension=1)
    fixed_z_ = torch.randn(4, 1024)  # fixed noise
    result = generator(fixed_z_)
    print(result.shape)
    pic = torch.ones(4, 3, 128, 64)
    pic = discriminator(pic)
    print(pic.shape)
