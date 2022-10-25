# coding: utf-8

import os
import torch
import torch.nn as nn
import collections
from model_aux import *
from models.alexnet import alexnet1d
from models.dla import dla34
from models.squeezenet import squeezenet
from models.shufflenetv2 import shufflenetv2
import numpy as np
import torchvision.models.squeezenet

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# d = np.resize(a, new_shape=(1, 33))
# print(np.max(d), np.min(d))
# print(d)

if __name__ == "__main__":

    """
    model = alexnet1d(14)
    # 加载预训练模型
    weights = r'./weights/epoch_610_acc_0.9985786080360413_accconf_0.9976606455433817.pt'
    if os.path.exists(weights):
        checkpoint = torch.load(weights, map_location='cpu')
        if 'model' in checkpoint:
            print('resume statedict...')
            model.load_state_dict(checkpoint['model'])

    x = torch.rand(size=(1, 1, 124), dtype=torch.float32, requires_grad=True)
    onnx_export_name = 'alex_1d.onnx'
    torch.onnx.export(model, x, onnx_export_name, input_names=['data'], output_names=['output'], verbose=False, opset_version=9)
    """

    """
    model = squeezenet(num_classes=14)
    model.eval()

    # 加载预训练模型
    weights = r'./weights/latest.pt'
    if os.path.exists(weights):
        checkpoint = torch.load(weights, map_location='cpu')
        if 'model' in checkpoint:
            print('resume statedict...')
            # model.load_state_dict(checkpoint['model'])

    b = 32
    x = torch.rand(size=(b, 1, 124), dtype=torch.float32)
    output = model(x)
    print('squeezenet output: ', output.shape)
    onnx_export_name = 'squeezenet{}.onnx'.format(b)
    torch.onnx.export(model, x, onnx_export_name, input_names=['data'], output_names=['output'], verbose=False,
                          opset_version=9)
    """


    model = dla34(num_classes=14)
    model.eval()

    # 加载预训练模型
    weights = r'./weights/epoch_0_acc_0.9845014810562134_accconf_0.0.pt'
    if os.path.exists(weights):
        checkpoint = torch.load(weights, map_location='cpu')
        if 'model' in checkpoint:
            print('resume statedict...')
            model.load_state_dict(checkpoint['model'])

    b = 1
    x = torch.rand(size=(b, 1, 124), dtype=torch.float32)
    onnx_export_name = 'dla34_1d_b{}.onnx'.format(b)
    torch.onnx.export(model, x, onnx_export_name, input_names=['data'], output_names=['output'], verbose=False,
                          opset_version=9)

