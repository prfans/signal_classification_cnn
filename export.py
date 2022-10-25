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

if __name__ == "__main__":
    model = dla34(num_classes=14)
    model.eval()

    # 加载预训练模型
    weights = r'./weights/xxx.pt'
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

