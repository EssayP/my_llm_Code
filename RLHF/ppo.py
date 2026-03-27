import torch
import torch.nn as nn
import math

class PPO(nn.Module):
    def __init__(self,clip=0.2,gamma=1.0,lam=0.95):
        self.clip = clip
        self.gamma = gamma
        self.lam = lam

    def mask_mean(self,loss,mask,dim =-1):
        return (loss * mask)