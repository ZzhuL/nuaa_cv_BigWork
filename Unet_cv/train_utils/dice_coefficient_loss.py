# -*- encoding: utf-8 -*-
"""
    @Project: Unet_cv.py
    @File   : dice_coefficient_loss.py
    @Author : ZHul
    @E-mail : zl2870@qq.com
    @Data   : 2023/5/13  20:47
"""
import torch
#import torchmetrics
import torch.nn as nn
import numpy as np


def build_target(target: torch.Tensor, num_classes: int = 3, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    # print("-----------------------------------")
    #print(dice_target.shape)
    if ignore_index >= 0:       # 将像素为255转为0
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        #dice_target = nn.functional.one_hot(dice_target.to(torch.int64), num_classes).float()
        dice_target[ignore_mask] = ignore_index
    # else:
    #     dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    #print("--------------------")
    # print(dice_target.permute(0, 3, 1, 2))
    #print(dice_target.shape)
    return dice_target # 将Chanel维度的数据移到索引为1的位置上来


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient

    d = 0.
    batch_size = x.shape[0]
    # print(x.shape)
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        # print(x_i.shape)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        if (channel!=0):  # 摘出肿瘤通道
            dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)
    # tumor通道单独计算
    d=0.
    isTumorNums =0
    x_isTumor=x[:, 0, :, :]
    target_isTumor=target[:, 0, :, :]
    for i in range(x_isTumor.shape[0]):
        x_i = x_isTumor[i].reshape(-1)
        t_i = target_isTumor[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter
        if ((x_i==1).any()):  # 只计算含有肿瘤的通道
            d += (2 * inter + epsilon) / (sets_sum + epsilon)
            isTumorNums+=1

    if(isTumorNums!=0):
        dice+=(d / isTumorNums)
        return dice / x.shape[1]

    return dice /( x.shape[1]-1) # 如果整个batchsize为0 的话


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

