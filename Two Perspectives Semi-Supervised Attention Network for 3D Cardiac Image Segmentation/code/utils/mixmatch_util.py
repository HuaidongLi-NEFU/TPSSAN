import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import random


def augmentation_torch(volume, aug_factor):
    # volume is numpy array of shape (C, D, H, W)
    noise = torch.clip(torch.randn(*volume.shape) * 0.1, -0.2, 0.2).cuda()
    return volume + aug_factor * noise#.astype(np.float32)

def mix_module(X, U, eval_net, K, T,alpha, mixup_mode, aug_factor):
    X_b = len(X)
    U_b = len(U)

    # step 1: Augmentation with random noise
    # aug_factor = torch.tensor(aug_factor)
    X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        Y_u,_,_,_ = eval_net(U_cap)  #the model now is four output, and the teacher network only use the first output
        Y_u = F.softmax(Y_u, dim=1)
        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)
    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # to get the two experience average result

    guessed = guessed.repeat(K, 1, 1, 1, 1)
    guessed = torch.argmax(guessed, dim=1)
    pseudo_label = guessed      #get the pseudo label

    U_cap = list(zip(U_cap, guessed))  # Merge pseudo labels and augumented data

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == '_':
        X_prime = X_cap
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'x':
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [cutmix(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    else:
        raise ValueError('wrong mixup_mode')

    return X_prime, U_prime, pseudo_label

def cutmix(s1, s2, alpha):
    x1, p1 = s1
    x2, p2 = s2

    # 图像大小，假设这里是 (C, H, W)
    _, d, h, w = x1.shape

    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    # 随机选择裁剪框的位置和大小
    rand_h = np.random.randint(0, h)
    rand_w = np.random.randint(0, w)
    cut_h = int(h * np.sqrt(1 - l))
    cut_w = int(w * np.sqrt(1 - l))

    # 创建裁剪区域的掩码
    mask = np.zeros((h, w))
    mask[rand_h:rand_h+cut_h, rand_w:rand_w+cut_w] = 1

    # 用 s2 的裁剪区域替换 s1
    x1[:, rand_h:rand_h+cut_h, rand_w:rand_w+cut_w] = x2[:, rand_h:rand_h+cut_h, rand_w:rand_w+cut_w]

    # 计算标签的混合
    p = l * p1 + (1 - l) * p2

    return (x1, p)


