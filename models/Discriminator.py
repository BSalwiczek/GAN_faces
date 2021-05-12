from torch import nn
from models.utils import Print

class Discriminator(nn.Module):
    def __init__(self, architecture, input_dim):
        super().__init__()

        if architecture == 0:
            in_channels = 3
            channels = 32
            self.main = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.Conv2d(in_channels, channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels) x 64 x 64
                nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*2) x 32 x 32
                nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*4) x 16 x 16
                nn.Conv2d(channels * 4, channels * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*8) x 8 x 8
                nn.Conv2d(channels * 8, channels * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*16) x 4 x 4
                nn.Conv2d(channels * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        if architecture == 1:
            self.main = nn.Sequential(
                nn.Conv2d(3, 48, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(48),

                nn.Conv2d(48, 96, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(96),

                nn.Conv2d(96, 192, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(192),

                nn.Conv2d(192, 384, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(384),

                nn.Conv2d(384, 768, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(768),

                nn.Conv2d(768, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()

            )

        if architecture == 3:
            in_channels = 3
            channels = 16
            self.main = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.Conv2d(in_channels, channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels) x 64 x 64
                nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*2) x 32 x 32
                nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels * 4, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*4) x 16 x 16
                nn.Conv2d(channels * 4, channels * 8, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels * 8, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*8) x 8 x 8
                nn.Conv2d(channels * 8, channels * 16, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels * 16, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (channels*16) x 4 x 4
                nn.Conv2d(channels * 16, 1, 4, 1, 0, bias=False),
            )

    def forward(self, x):
        return self.main(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
