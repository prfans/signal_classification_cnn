# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, loss_fcn=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num
        if loss_fcn is not None:
            self.loss_fcn = loss_fcn
        else:
            if self.class_num == 1:
                self.loss_fcn = torch.nn.BCEWithLogitsLoss()
                self.loss_fcn.reduction = 'none'  # required to apply FL to each element
            else:
                self.loss_fcn = torch.nn.CrossEntropyLoss()
                self.loss_fcn.reduction = 'none'

    def forward(self, predict, target):
        if self.class_num > 1:
            pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
            class_mask = F.one_hot(target, self.class_num)  # 获取target的one hot编码
            ids = target.view(-1, 1)
            alpha = self.alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor#),里面的元素分别是每一个类的权重因子
            probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt，每个样本的概率
            # log_p = -probs.log()
            log_p = self.loss_fcn(predict, target)
            loss = alpha.to(probs.device) * (torch.pow((1 - probs), self.gamma)) * log_p  # 原始ce上增加一个动态权重衰减因子
        else:
            loss = self.loss_fcn(predict, target)
            pred_prob = torch.sigmoid(predict)  # prob from logits
            p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            modulating_factor = (1.0 - p_t) ** self.gamma
            loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

if __name__ == "__main__":
    pass
