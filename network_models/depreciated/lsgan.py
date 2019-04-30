import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Generator_(nn.Module):
    def __init__(self, nz,  nChannels, ngf=64):
        super(Generator_, self).__init__()
        # input : z
        # Generator will be consisted with a series of deconvolution networks
        self.nz = nz
        self.layer1 = nn.Sequential(
            # input : z
            # Generator will be consisted with a series of deconvolution networks

            # Input size : input latent vector 'z' with dimension (nz)*1*1
            # Output size: output feature vector with (ngf*8)*4*4
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            # Input size : input feature vector with (ngf*8)*4*4
            # Output size: output feature vector with (ngf*4)*8*8
            nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            # Input size : input feature vector with (ngf*4)*8*8
            # Output size: output feature vector with (ngf*2)*16*16
            nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        self.layer4 = nn.Sequential(
            # Input size : input feature vector with (ngf*2)*16*16
            # Output size: output feature vector with (ngf)*32*32
            nn.ConvTranspose2d(
                in_channels=ngf * 2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        self.layer5 = nn.Sequential(
            # Input size : input feature vector with (ngf)*32*32
            # Output size: output image with (nChannels)*(image width)*(image height)
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=nChannels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()  # To restrict each pixels of the fake image to 0~1
            # Yunjey seems to say that this does not matter much
        )

    def forward(self, x):
        x = x.view(-1, self.nz, 1, 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


class Discriminator_(nn.Module):
    def __init__(self, nChannels, ndf=64):
        super(Discriminator_, self).__init__()
        # input : (batch * nChannels * image width * image height)
        # Discriminator will be consisted with a series of convolution networks

        self.layer1 = nn.Sequential(
            # Input size : input image with dimension (nChannels)*64*64
            # Output size: output feature vector with (ndf)*32*32
            nn.Conv2d(
                in_channels=nChannels,
                out_channels=ndf,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            # Input size : input feature vector with (ndf)*32*32
            # Output size: output feature vector with (ndf*2)*16*16
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            # Input size : input feature vector with (ndf*2)*16*16
            # Output size: output feature vector with (ndf*4)*8*8
            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            # Input size : input feature vector with (ndf*4)*8*8
            # Output size: output feature vector with (ndf*8)*4*4
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 4,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            # Input size : input feature vector with (ndf*8)*4*4
            # Output size: output probability of fake/real image
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 4,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Linear(in_features=256 * 8, out_features=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(-1, 256 * 4 * 2)
        out = self.linear(out)

        return out


class Discriminator(nn.Module):
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

    def __init__(self, input_channels, output_dimension=1, gpu_ids=None):
        super(Discriminator, self).__init__()
        self.model = Discriminator_(input_channels)
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
    net = Generator(
        input_dimension=100, output_channels=3
    )
    print "Input(=z) : ",
    print(torch.randn(128,100).size())
    y = net(Variable(torch.randn(128,100))) # Input should be a 4D tensor
    print "Output(batchsize, channels, width, height) : ",
    print(y.size())