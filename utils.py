# coding: utf-8

import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from augs import Compose, RandomBlur, Normal, ToTensor, RandomNoise, RandomCrop
from dataset import TxtFolder

def get_network(args):
    """ return given network
    """

    print("==> creating model '{}'".format(args.net))
    print("==> class_num {}".format(args.class_num))

    if args.net == 'alexnet':
        from models.alexnet import alexnet1d
        net = alexnet1d(num_classes=args.class_num)
    elif args.net == 'dla34':
        from models.dla import dla34
        net = dla34(num_classes=args.class_num)
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(num_classes=args.class_num)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(num_classes=args.class_num)
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(num_classes=args.class_num)
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(num_classes=args.class_num)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(num_classes=args.class_num)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=args.class_num)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=args.class_num)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=args.class_num)

    else:
        print('the network name you have entered ({}) is not supported yet'.format(args.net))
        sys.exit()

    return net


def get_training_dataloader(path, args, hyp):
    """ return training dataloader
    """

    transform_train = Compose([
        RandomBlur(),
        RandomNoise(),
        RandomCrop(20),
        Normal(args.image_size),
        ToTensor(),
    ])

    train_set = TxtFolder(img_paths=path, img_transform=transform_train,
                          class_num=args.class_num, max_data_num_c=args.max_data_num_c)
    data_loader = DataLoader(train_set, shuffle=args.shuffle, num_workers=args.num_workers,
                             batch_size=args.b, drop_last=args.drop_last)

    return data_loader


def get_test_dataloader(path, args, hyp):
    """ return training dataloader
    """

    transform_test = Compose([
        Normal(args.image_size),
        ToTensor(),
     ])

    print('##########################load images############################')
    test_set = TxtFolder(img_paths=path, img_transform=transform_test,
                         class_num=args.class_num, max_data_num_c=args.max_data_num_c)
    data_loader = DataLoader(test_set, shuffle=args.shuffle, num_workers=args.num_workers, batch_size=args.b)

    return data_loader

if __name__ == "__main__":
    import numpy as np
    import torch

    data = np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32)
    print(data, data.shape)

    data = np.resize(data, new_shape=(1, 120))
    print(data, data.shape)
