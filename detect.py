# coding: utf-8


import torch
import numpy as np
import os
import cv2
import shutil
from model_aux import *
from models.alexnet import alexnet1d
from models.dla import dla34
from models.squeezenet import squeezenet
from models.shufflenetv2 import shufflenetv2
from get_file_list import get_file_list
from augs import Compose, Normal, ToTensor
import matplotlib.pyplot as plot


txt_paths = [
    r'data',
]

pre_process = Compose([
    Normal(124),
    ToTensor(),
])


def load_txt(file):
    with open(file, 'r') as fd:
        lines = fd.readlines()
        data = np.array([x.split(',') for x in lines]).astype(dtype=np.float32)
    return data


def plot_txt(file, image_file):
    name = path.split('\\')[-1].split('/')[-1]
    index = path.split('\\')[-2]

    data = load_txt(file)

    x = [i for i in range(0, len(data[0]))]
    y = data[0]

    plot.plot(x, y, color='red', label='class-{} / {}'.format(index, name))
    plot.legend(loc='upper right')
    plot.savefig(fname=image_file, figsize=[10, 10])
    plot.close()


save_dir_failed = r'data'
save_dir_ok = r'data1'

if not os.path.exists(save_dir_failed):
    os.mkdir(save_dir_failed)
if not os.path.exists(save_dir_ok):
    os.mkdir(save_dir_ok)

if __name__ == "__main__":
    model = dla34(num_classes=14)
    model.eval()

    # 加载预训练模型
    weights = r'./weights/xxx.pt'
    if os.path.exists(weights):
        checkpoint = torch.load(weights, map_location='cpu')
        if 'model' in checkpoint:
            print('resume form {}'.format(weights))
            model.load_state_dict(checkpoint['model'])

    len_total = 0
    failed = 0
    for txt_path in txt_paths:
        txts = get_file_list(txt_path, '.txt')
        len_total += len(txts)
        for path in txts:
            print(path, end=' ')
            txt_name = path.split('\\')[-1].split('/')[-1]
            cls_id = path.split('\\')[-2]
            im_path = path.replace('.txt', '.png')

            # 读取并推理
            data = load_txt(path)
            data = pre_process(data)
            data = torch.unsqueeze(data, 0)
            output = model(data)
            _, pred = output.max(1)

            output_ = torch.nn.functional.softmax(output, 1)
            scores = output_.detach().numpy()
            score = scores.max(1)[0]
            pred = pred.cpu().numpy()[0]

            print(pred, score)

            if score > 0.98:
                save_path = os.path.join(save_dir_ok, str(pred))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                # new_name = str(pred) + '_' + txt_name
                dst_file = os.path.join(save_path, txt_name)
                shutil.copyfile(path, dst_file)
            else:
                save_path = os.path.join(save_dir_failed, cls_id)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                dst_name = '{}_{}_{}'.format(pred, score, txt_name)
                dst_file = os.path.join(save_path, dst_name)
                shutil.copyfile(path, dst_file)

                plot_txt(dst_file, dst_file.replace('.txt', '.png'))

                failed += 1

    print('total {}, failed {}'.format(len_total, failed))

