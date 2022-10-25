# coding: utf-8
""" image auguments, cv2 and IPL.Image"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import torchvision
import cv2


class Compose(list):
    """ This is lightnet's own version of :class:`torchvision.transforms.Compose`.

    Note:
        The reason we have our own version is because this one offers more freedom to the user.
        For all intends and purposes this class is just a list.
        This `Compose` version allows the user to access elements through index, append items, extend it with another list, etc.
        When calling instances of this class, it behaves just like :class:`torchvision.transforms.Compose`.

    Note:
        I proposed to change :class:`torchvision.transforms.Compose` to something similar to this version,
        which would render this class useless. In the meanwhile, we use our own version
        and you can track `the issue`_ to see if and when this comes to torchvision.
    """

    def __call__(self, data):
        for tf in self:
            data = tf(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ['
        for tf in self:
            format_string += '\n  {tf}'
        format_string += '\n]'
        return format_string


# 随机模糊
class RandomBlur(object):
    def __init__(self, kernel=[0.1, 0.8, 0.1]):
        self.kernel = kernel

    def __call__(self, data):
        if random.randint(0, 10) >= 8:
            shape = data.shape
            data = data.reshape((shape[1],))
            data = np.convolve(data, self.kernel, 'same').reshape(shape)
        return np.float32(data)


# 归一化操作
class Normal(object):
    def __init__(self, size=120):
        self.normal_size = size

    def __call__(self, data):

        # data = 2 * (np.resize(data, new_shape=(1, self.normal_size)) - np.min(data)) / (np.max(data) - np.min(data)) - 1

        # """
        shape = max(data.shape)
        data_0 = np.zeros(shape=(1, self.normal_size))
        data_0[:, :min(shape, self.normal_size)] = data[:, :min(shape, self.normal_size)]
        data = data_0
        min_v = np.min(data)
        max_v = np.max(data)
        data = 2 * (data - min_v) / (max_v - min_v) - 1
        #"""

        return np.float32(data)


# 转换tensor
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return torch.from_numpy(data)


# 随机加噪声
class RandomNoise(object):
    def __init__(self):
        pass

    def __call__(self, data):
        if random.randint(0, 10) >= 8:
            noise = np.random.uniform(-1.0, 1.0, len(data))
            data = data + noise
        return np.float32(data)


# 随机剪裁
class RandomCrop(object):
    def __init__(self, crop_size=10):
        self.crop_size = crop_size # 平移范围

    def __call__(self, data):
        if self.crop_size >= len(data):
            return data

        if random.randint(0, 10) >= 9:
            p = random.randint(0, self.crop_size)
            left_or_right = random.randint(0, 10)
            if left_or_right < 5:
                data = data[p:]
            else:
                if p > 0:
                    data = data[:-p]
        return np.float32(data)


if __name__ == '__main__':

    data = np.array([1, 2, 3, 4, 5, -1, -2, -3, -4, -5], dtype=np.float32)
    print(data, data.shape)

    norm = Normal(5)
    output = norm(data)
    print(output, output.shape)

