from torch import nn
from models.utils import Print


class Generator(nn.Module):
    def __init__(self, architecture, input_dim, z_dim):
        super().__init__()

        if architecture == 0 or architecture == 1 or architecture == 3:
            channels = 16
            out_channels = 3
            self.main = nn.Sequential(
                nn.ConvTranspose2d(z_dim, channels * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(channels * 16),
                nn.ReLU(True),

                nn.ConvTranspose2d(channels * 16, channels * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(channels * 8, channels * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels*2),
                nn.ReLU(True),

                nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),

                nn.ConvTranspose2d(channels, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, x):
        return self.main(x)
