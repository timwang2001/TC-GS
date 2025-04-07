#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def wavelet_loss(img1,img2,step=0):
    xfm = DWTForward(J=2, mode='zero', wave='haar').to('cuda')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='haar')
    Yl1, Yh1 = xfm(img1.unsqueeze(0))
    Yl2, Yh2 = xfm(img2.unsqueeze(0))
    Ll = l1_loss(Yl1.squeeze(),Yl2.squeeze())
    Yh11 = Yh1[0].permute(2,0,1,3,4).squeeze()#3,c,h,w
    Yh21 = Yh2[0].permute(2,0,1,3,4).squeeze()
    Yh12 = Yh1[1].permute(2,0,1,3,4).squeeze()
    Yh22 = Yh2[1].permute(2,0,1,3,4).squeeze()

    lmbda1,lmbda2 = 1.0-step/30000, step/30000
    if step >25000:
        loss = lmbda1*(0.4 * l1_loss(Yh11[0],Yh21[0]) + 0.3 *l1_loss(Yh11[1],Yh21[1]) + 0.3 *l1_loss(Yh11[2],Yh21[2]))+lmbda2*( 0.4 * l1_loss(Yh12[0],Yh22[0]) + 0.3 *l1_loss(Yh12[1],Yh22[1]) + 0.3 *l1_loss(Yh12[2],Yh22[2]))

    else :
        loss = lmbda1*Ll +lmbda2*(0.2*l1_loss(Yh11[0],Yh21[0]) + 0.2 *l1_loss(Yh11[1],Yh21[1]) + 0.2 *l1_loss(Yh11[2],Yh21[2])+0.2 * l1_loss(Yh12[0],Yh22[0]) + 0.2 *l1_loss(Yh12[1],Yh22[1]) + 0.2 *l1_loss(Yh12[2],Yh22[2]))
    return loss
