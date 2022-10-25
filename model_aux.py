# coding: utf-8

import os
import torch
import collections


@torch.no_grad()
def weight_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
        # m.bias.fill_(0.0)

    elif type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # torch.nn.init.normal_(m.weight, 0.0, 0.1)

    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)


def init_model_parameters(model):
    model.apply(weight_init)


def init_model(model):
    return init_model_parameters(model)


def save_model(model, model_file):
    if isinstance(model, collections.OrderedDict) or isinstance(model, dict):
        torch.save(model, model_file)
    elif isinstance(model, torch.nn.Module):
        state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save(state_dict, model_file)
    else:
        raise RuntimeError('type:{} not support'.format(type(model)))


def load_state_dict(model, state):
    if isinstance(state, str):
        if not os.path.exists(state):
            raise RuntimeError('{} not exist'.format(state))
        state_dict = torch.load(state)
        model.load_state_dict(state_dict)
    elif isinstance(state, collections.OrderedDict):
        model.load_state_dict(state)
    else:
        raise RuntimeError('type:{} not support'.format(type(state)))


def load_model(model, state):
    return load_state_dict(model, state)


