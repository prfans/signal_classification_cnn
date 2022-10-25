# coding: utf-8

import os
import sys
import time
import numpy as np
from pathlib import Path
import random
import yaml
import torch
import torch.nn as nn
import argparse
from torch.optim import lr_scheduler

from model_aux import save_model
from utils import get_network, get_training_dataloader, get_test_dataloader
from FocalLoss import FocalLoss

# 模型训练
def train_epoch(net, loss_function, optimizer, training_loader, epoch):
    """ train epoches """

    total_loss = 0.0
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if not args.no_gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()

        outputs = net(images)
        #print(outputs.shape, labels.shape, labels)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()


        total_loss += loss.item()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    return total_loss


# 模型测试
@torch.no_grad()
def eval_training(net, loss_function, test_loader, epoch, conf):
    """ eval function """

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    correct_conf = 0.0

    total_image = 0

    for (images, labels) in test_loader:

        if not args.no_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        total_image += images.shape[0]

        #
        outputs_ = torch.nn.functional.softmax(outputs, 1)
        scores = outputs_.cpu().numpy()
        scores = scores.max(1)
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        correct_conf += (scores[preds == labels] > conf).sum()

    finish = time.time()
    #if args.use_gpu:
    #    print('GPU INFO.....')
    #    print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Total: {:.4f}, ErrorNum: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Accuracy Conf: {:.4f}, Time consumed:{:.2f}s'.format(
        total_image,
        total_image - correct,
        test_loss / total_image,
        correct.float() / total_image,
        float(correct_conf) / total_image,
        finish - start
    ))
    print()

    return correct.float() / total_image, correct.int(), float(correct_conf) / total_image, int(correct_conf), test_loss


# 超参优化每次实验过程
def train_and_eval(hyp):
    """ train function """

    model = get_network(args)

    # 数据处理
    training_loader = get_training_dataloader(args.train, args, hyp)
    test_loader = get_test_dataloader(args.test, args, hyp)

    # resume
    latest_pt = args.weights
    if args.resume and os.path.exists(latest_pt):
        print('resume from {}....'.format(latest_pt))
        checkpoint = torch.load(latest_pt, map_location='cpu')

        if 'epoch' in checkpoint:
            print('resume epoch...', )
            resume_epoch = checkpoint['epoch'] + 1

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = torch.nn.DataParallel(model)
        model.cuda().train()

        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
        if 'optimizer' in checkpoint:
            print('resume optimizer...')
            optimizer.load_state_dict(checkpoint['optimizer'])

        # lr_sch = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1*epoch, last_epoch=start_epoch)
        # lr_sch = lr_scheduler.StepLR(optimizer, 1, 0.1, last_epoch=start_epoch)
        # lr_sch = lr_scheduler.MultiStepLR(optimizer, [2, 4, 6], 0.1, last_epoch=start_epoch)
        # lr_sch = lr_scheduler.CosineAnnealingLR(optimizer, 5, 0.000001, last_epoch=start_epoch)
        lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=50, verbose=True)
        if 'lr_scheduler' in checkpoint:
            pass
            # print('resume lr_scheduler...')
            # lr_sch.load_state_dict(checkpoint['lr_scheduler'])

        del checkpoint  # current, saved
    else:
        resume_epoch = 0
        model.cuda().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
        lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=50, verbose=True)

    if not args.no_gpu:
        if len(args.gpu) > 1:
            model = torch.nn.DataParallel(model)
            # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
            #torch.distributed.init_process_group(backend="nccl")
            #net = torch.nn.parallel.DistributedDataParallel(net)
        model = model.cuda()

    # 训练参数设置
    # loss_function = nn.CrossEntropyLoss()
    loss_function = FocalLoss(class_num=14, gamma=2)
    # 如果用center_loss则center_loss的参数需要更新

    test_best_acc = 0.0
    test_best_acc_conf = 0.0

    # train
    for epoch in range(resume_epoch, args.EPOCH):
        loss_train = train_epoch(model, loss_function, optimizer, training_loader, epoch)
        acc, correct_num, acc_conf, correct_conf_num, test_loss = eval_training(model, loss_function, test_loader, epoch, hyp['conf'])
        print('acc={}, acc_conf={}, test_loss={}'.format(acc, acc_conf, test_loss))

        checkpoint = {'epoch': epoch,
                      'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_sch.state_dict()}
        save_path = os.path.join(args.save_folder, 'latest.pt')
        save_model(checkpoint, save_path)


        if test_best_acc < acc:
            test_best_acc = acc
            save_path = os.path.join(args.save_folder, 'epoch_{}_acc_{}_accconf_{}.pt'.format(epoch, acc, acc_conf))
            save_model(checkpoint, save_path)
        if test_best_acc_conf < acc_conf:
            test_best_acc_conf = acc_conf
            save_path = os.path.join(args.save_folder, 'epoch_{}_acc_{}_accconf_{}.pt'.format(epoch, acc, acc_conf))
            save_model(checkpoint, save_path)

        lr_sch.step(acc_conf)


        # 普通训练及测试
def train_and_eval_args(opts, hyp):
    global args, best_weight
    args = opts
    best_weight = args.evolve_weight
    args.drop_last = False
    return train_and_eval(hyp)


# load Hyperparameters
def load_hyperparamerters(opts):
    with open(opts.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    return hyp


def printf_train_test_sets(args):
    print('==>train_sets: ')
    for set in args.train:
        print(' ', set)
    print()
    print('==>test_sets:')
    for set in args.test:
        print(' ', set)
    print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="dla34", required=False, help='net type')

    parser.add_argument('-EPOCH', type=int, default=10000, help='EPOCH')
    parser.add_argument('-no_gpu', action='store_true', help='use gpu or not')
    parser.add_argument('-gpu', type=str, default='0', help='gpus')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-resume', action='store_true', help='resume training')
    parser.add_argument('-weights', type=str, default="./weights/latest.pt", required=False, help='weights')
    parser.add_argument('-save_folder', type=str, default="./weights", required=False, help='save_folder')

    parser.add_argument('-num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('-no_shuffle', action='store_true', help='shuffle')
    parser.add_argument('-hyp', type=str, default=r"./hyp.yaml", help='Hyperparameters file')
    parser.add_argument('-class_num', type=int, default=14, help='class_num')
    parser.add_argument('-image_size', type=int, default=124, required=False, help='image_size')
    parser.add_argument('-cascade', type=str, required=False, default=None, help='cascade txt')
    parser.add_argument('-save_dir', type=str, required=False, default='output', help='output')
    parser.add_argument('-hypopt', type=str, required=False, default=None, help='hyperparameters optimizer')
    parser.add_argument('-max_data_num_c', type=int, default=-1, help='max sample number per class')
    parser.add_argument('-ho_search_num', type=int, default=3000, help='hyper optimizer search number')
    parser.add_argument('-max_loss', type=float, default=100.0, help='max loss')

    # 模型参数微调时使用的参数文件
    evolve_weight = None
    parser.add_argument('-evolve_weight', type=str, default=evolve_weight, help='evolve_weight')
    parser.add_argument('-update_evolve', action='store_true', help='update_evolve')

    # 训练集和测试集
    default_trains = [r"/data1/", r"/data2"]
    default_tests = [r"/data3/", r"/data4"]
    parser.add_argument('-train', type=str, default=default_trains, help='train set')
    parser.add_argument('-test', type=list, default=default_tests, help='test sets')

    args = parser.parse_args()

    # 输出train and test sets
    printf_train_test_sets(args)

    # 设置gpu
    if not args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print('==> hyperparameters: ', args.hyp)

    # 加载超参数Hyperparameters
    hyp = load_hyperparamerters(args)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    train_and_eval_args(args, hyp)

