# coding: utf-8

import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from get_file_list import *


# 加载完毕后默认都是pytorch的tensor
class TxtFolder(Dataset):
    def __init__(self, img_paths, img_transform=None, class_num=1, max_data_num_c=-1, noraml_size=120):
        self.img_transform = img_transform
        self.class_num = class_num
        self.img_list = []
        self.label_list = []
        self.ext = '.txt'
        self.noraml_size = noraml_size
        for img_path in img_paths:
            if self.class_num == 1:
                self.img_list += get_file_list(img_path, self.ext)
                self.label_list += [0 for i in range(len(self.img_list))]
            else:
                for c in range(self.class_num):
                    c_path = os.path.join(img_path, str(c))
                    objs_num_c_path = 0
                    if os.path.exists(c_path):
                        c_img_list = get_file_list(c_path, self.ext)
                        if max_data_num_c != -1 and len(c_img_list) > max_data_num_c:
                            c_img_list = c_img_list[:max_data_num_c]
                        self.img_list += c_img_list
                        objs_num_c_path = len(c_img_list)
                        self.label_list += [c for i in range(objs_num_c_path)]
                    print(' #{}: {} has {} objs'.format(c, c_path, objs_num_c_path))
                print('=================================================')


    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        image_data = self.load_path_images(img_path)
        while image_data is None: # image is not existing
            print('{} is None'.format(img_path))
            index = index + 1
            img_path = self.img_list[index]
            label = self.label_list[index]
            image_data = self.load_path_images(img_path)

        if self.img_transform is not None:
            image_data = self.img_transform(image_data)

        # print('img_path ', img_path, ' label ', label)

        return image_data, label

    def load_path_images(self, image_path):
        if not os.path.exists(image_path):
            return None
        with open(image_path, 'r') as fd:
            lines = fd.readlines()
            data = np.array([x.split(',') for x in lines]).astype(dtype=np.float32)
            # data = 2* (np.resize(data, new_shape=(1, self.noraml_size)) -np.min(data)) / (np.max(data)-np.min(data)) - 1
            # data = torch.from_numpy(data)
        return data

    def __len__(self):
        return len(self.img_list)

    def size(self):
        return len(self.img_list)


if __name__ == '__main__':

    data_root = [r'D:\data\tem_MICR\white_1']

    # ####################################################
    # test PIL.Image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = TxtFolder(data_root, class_num=14)
    print(len(dataset), dataset[0][0], np.min(dataset[0][0]), np.max(dataset[0][0]), dataset[0][1])

    #data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    #for i, d in enumerate(data_loader):
    #    print('i= ', i, d[0].shape, d[1])