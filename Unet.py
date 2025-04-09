import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(DoubleConv, self).__init__()
        self.double_conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, incept=False):
        super(DownSample, self).__init__()
        if incept:
            self.conv = InceptionBlock(in_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels, out_channels, k)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, incept=False):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        if incept:
            self.conv = InceptionBlock(in_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class InceptionBlock(nn.Module):
    """
    Inception block with 4 branches.
    """
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        branch_channels = out_channels // 4  # Divide output channels among branches

        self.conv1x1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)

        self.double_conv3x3_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv5x5_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv3x3_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Reduces dimensionality: Ensures that the output has the correct number of channels (out_channels), even if the concatenation resulted in more channels.
        self.conv1x1_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Merge output channels

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.double_conv3x3_3x3(x)
        x3 = self.double_conv5x5_5x5(x)
        x4 = self.double_conv3x3_1x1(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv1x1_out(out)

class Unet(nn.Module):
    """
    Modular U-Net implementation.

    :param in_channels (int): Number of input channels.
    :param num_features (int): Number of output features (channels).
    :param depth (int): Number of downsampling/upsampling layers.
    :param base_channels (int): Number of channels in the first convolutional layer.
    :param k_size (int): Kernel size for convolutional layers.
    :param inception (boolean): Use inception blocks instead of double convolutions.
    """
    def __init__(self, in_channels, num_features, depth=4, base_channels=64, k_size=3, inception=False):

        super(Unet, self).__init__()
        self.depth = depth

        # Create downsampling layers
        self.down_layers = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.down_layers.append(DownSample(in_ch, out_ch, k=k_size, incept=inception))
            in_ch = out_ch

        # Bottleneck
        self.bottle_neck = DoubleConv(in_ch, in_ch * 2)

        # Create upsampling layers
        self.up_layers = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            self.up_layers.append(UpSample(in_ch * 2, out_ch, incept=inception))
            in_ch = out_ch // 2

        # Output layer
        self.out = nn.Conv2d(base_channels, out_channels=num_features, kernel_size=1)

    def forward(self, x):
        downs = []
        p = x

        # Apply downsampling
        for layer in self.down_layers:
            down, p = layer(p)
            downs.append(down)

        # Bottleneck
        b = self.bottle_neck(p)

        # Apply upsampling
        up = b
        for layer, down in zip(self.up_layers, reversed(downs)):
            up = layer(up, down)

        # Final output layer
        out = self.out(up)
        # return torch.sigmoid(out)
        return out
    
