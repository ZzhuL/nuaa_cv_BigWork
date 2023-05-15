# -*- encoding: utf-8 -*-
"""
    @Project: Unet_cv.py
    @File   : unet_ags.py
    @Author : ZHul
    @E-mail : zl2870@qq.com
    @Data   : 2023/5/15  18:55
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# class MaxPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(MaxPooling, self).__init__(
#             nn.MaxPool2d(2, stride=2)
#         )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        # diff_y = x2.size()[2] - x1.size()[2]
        # diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        # x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
        #                 diff_y // 2, diff_y - diff_y // 2])
        #
        # x = torch.cat([x2, x1], dim=1)
        # x = self.conv(x1)
        return x1


def Crop(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = self.up(x1)
    # [N, C, H, W]
    diff_y = x2.size()[2] - x1.size()[2]
    diff_x = x2.size()[3] - x1.size()[3]

    # padding_left, padding_right, padding_top, padding_bottom
    x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2])

    x = torch.cat([x2, x1], dim=1)
    x = self.conv(x)
    return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class AG_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 3,
                 bilinear: bool = False,
                 base_c: int = 64):
        super(AG_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.AG1 = Attention_block(base_c * 8, base_c * 8, base_c * 4)
        self.up1_conv = Down(base_c * 16, base_c * 8)

        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.AG2 = Attention_block(base_c * 4, base_c * 4, base_c * 2)
        self.up2_conv = Down(base_c * 8, base_c * 4)

        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.AG3 = Attention_block(base_c * 2, base_c * 2, base_c * 1)
        self.up3_conv = Down(base_c * 4, base_c * 2)

        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.AG4 = Attention_block(base_c, base_c, base_c // 2)
        self.up4_conv = Down(base_c * 2, base_c)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)

        x2 = self.Maxpool(x1)
        x2 = self.down1(x2)

        x3 = self.Maxpool(x2)
        x3 = self.down2(x3)

        x4 = self.Maxpool(x3)
        x4 = self.down3(x4)

        x5 = self.Maxpool(x4)
        x5 = self.down4(x5)

        d5 = self.up1(x5)
        x4 = self.AG1(d5, x4)
        # d5 = Crop(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up1_conv(d5)

        d4 = self.up2(d5)
        x3 = self.AG2(d4, x3)
        # d4 = Crop(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up2_conv(d4)

        d3 = self.up3(d4)
        x2 = self.AG3(d3, x2)
        # d3 = Crop(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up3_conv(d3)

        d2 = self.up4(d3)
        x1 = self.AG4(d2, x1)
        # d2 = Crop(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up4_conv(d2)

        # d1 = self.out_conv(d2)
        logits = self.out_conv(d2)

        return {"out": logits}
