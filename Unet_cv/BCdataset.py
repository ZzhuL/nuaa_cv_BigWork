# -*- encoding: utf-8 -*-
"""
    @Project: Unet_cv.py
    @File   : BCdataset.py
    @Author : ZHul
    @E-mail : zl2870@qq.com
    @Data   : 2023/5/13  20:51
"""

import torch
import os
import glob
import sys
import numpy as np
from torch.utils.data import Dataset
import hdf5storage
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_filename, img_name, mask_name):
        # 初始化函数，读取所有data_path下的图片
        self.mat_name = data_filename
        self.dataset = hdf5storage.loadmat(self.mat_name)
        self.train_img = self.dataset[img_name]
        self.img_width = np.size(self.train_img, 2)
        self.img_height = np.size(self.train_img, 1)

        self.train_label = self.dataset[mask_name]

    def __getitem__(self, index):
        image = self.train_img[index]
        label = self.train_label[index]
        label = label.transpose(2,0,1)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image)

        # image = image.dtype("float")

        return image, label

    def __len__(self):
        # 返回训练集大小
        self.train_img_num = np.size(self.train_img, 0)
        return self.train_img_num


if __name__ == "__main__":
    current_path = sys.path[0]
    train_data_filename = current_path + '\\data\\T2_train.mat'
    isbi_dataset = ISBI_Loader(train_data_filename, 'BC_train_img', 'BC_train_lab')
    # print(isbi_dataset.imgs_path)
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
