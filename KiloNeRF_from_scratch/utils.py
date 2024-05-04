import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from model import KiloNeRF

def calculate_pixel(c, sigma, x):
    delta = torch.norm(x[:,1:,:] - x[:,:-1,:], dim=-1).unsqueeze(-1)
    alpha = 1 - torch.exp(-sigma[:,:-1,:]*delta)
    T = torch.cumprod(1-alpha, dim=1)
    c_ = torch.sum(T*alpha*c[:,:-1,:], dim=1)
    return c_, T*alpha

def sample_pts(o, d, sample_set, bins_num=8):
    o_ = o.unsqueeze(-1).expand(o.shape[0], o.shape[1], bins_num).transpose(1,2) # [batch_size, bins_num, 3]
    d_ = d.unsqueeze(-1).expand(d.shape[0], d.shape[1], bins_num).transpose(1,2) # [batch_size, bins_num, 3]
    x = o_ + d_ * sample_set.unsqueeze(-1) # [batch_size, bins_num, 3]
    return x, d_

def render_pixel(model, xyz, pose, DEVICE='cuda', bmin=2, bmax=6, bins_num=192, ratio=1.0, scale = 1):
    # Create sample pts set
    sample_set = torch.linspace(bmin, bmax, bins_num).to(DEVICE)
    sample_set = sample_set.view(1, sample_set.shape[0]).expand(xyz.shape[0], bins_num) # [batch_size, bins_num]
    ss = (sample_set[:,1:] - sample_set[:,:-1]).to(DEVICE)
    ss = torch.cat((ss, ss[:,-1].unsqueeze(-1)), dim=1)

    r = ((torch.rand(sample_set.shape) * scale - scale / 2.0) * ratio).to(DEVICE)
    sample_set = torch.clamp(sample_set + ss * r, min=bmin, max=bmax)
    
    o, d = sample_pts(xyz, pose, sample_set, bins_num=bins_num) # 1024, 8, 3 # 1024, 8, 3
    color, sigma = model(o.reshape(-1, 3), d.reshape(-1, 3))
    
    color = color.reshape(o.shape)
    sigma = sigma.reshape((o.shape[0], o.shape[1], 1))
    c_, T = calculate_pixel(color, sigma, o)
    return c_ + 1 - torch.sum(T, dim=1)
