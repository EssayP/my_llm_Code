import torch
import torch.nn as nn
import math

class PPO(nn.Module):
    def __init__(self,clip=0.2,gamma=1.0,lam=0.95):
        super().__init__()
        self.clip = clip
        self.gamma = gamma
        self.lam = lam

    def mask_mean(self,loss,mask,dim =-1):
        return (loss * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)

    def advantage_estimate(self,rewards,values):
        batch_size, seq_len = rewards.size()
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in range(seq_len-1,-1,-1):
            next_value = values[:,t+1] if t < seq_len - 1 else 0.0
            delta = rewards[:,t] + self.gamma *next_value - values[:,t]
            gae = delta + self.gamma * self.lam * gae
            advantages[:,t] = gae

        returns = advantages + values
        return advantages,returns

    def policy_loss(self,new_log_probs, old_log_probs,advantages,act_masks):
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1- self.clip,1 + self.clip) * advantages
        loss = -torch.min(surr1,surr2)
        return self.mask_mean(loss,act_masks)

    def value_loss(self,new_values,returns,act_masks):
        loss = (new_values - returns) ** 2
        return self.mask_mean(loss,act_masks)