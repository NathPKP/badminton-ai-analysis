"""TrackNetV3 network definitions (PyTorch), used only to convert the published checkpoints to ONNX.

Architecture copied from qaz812345/TrackNetV3 (model.py, MIT licence, (c) 2024 qaz812345), which also
covers the pretrained weights. Layer names must match the checkpoints, so they are kept as published
(including the "buttleneck" spelling). Inference itself runs on the ONNX files with onnxruntime
(s13_shuttle.py), so PyTorch is not needed to analyse a video.
"""
import torch
from torch import nn


class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    """U-Net: (N, (L+1)*3, 288, 512) frames (median background first) -> (N, L, 288, 512) heatmaps."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pool, up = nn.MaxPool2d(2, 2), nn.Upsample(scale_factor=2)
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(pool(x1))
        x3 = self.down_block_3(pool(x2))
        x = self.bottleneck(pool(x3))
        x = self.up_block_1(torch.cat([up(x), x3], dim=1))
        x = self.up_block_2(torch.cat([up(x), x2], dim=1))
        x = self.up_block_3(torch.cat([up(x), x1], dim=1))
        return self.sigmoid(self.predictor(x))


class Conv1DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding="same", bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class Double1DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class InpaintNet(nn.Module):
    """1-D U-Net: (N, L, 2) normalised coordinates + (N, L, 1) mask -> (N, L, 2) coordinates."""

    def __init__(self):
        super().__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2).permute(0, 2, 1)
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x = self.buttleneck(x3)
        x = self.up_1(torch.cat([x, x3], dim=1))
        x = self.up_2(torch.cat([x, x2], dim=1))
        x = self.up_3(torch.cat([x, x1], dim=1))
        return self.sigmoid(self.predictor(x)).permute(0, 2, 1)
