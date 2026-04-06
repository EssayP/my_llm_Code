import torch
import torch.nn as nn

class GRPO:
    def __init__(self,clip,eps,beta):
        self.clip = clip
        self.eps = eps
        self.beta = beta

    def group_advantages(self,rewards,group_size):
        B = rewards.shape[0]
        G = B // group_size
        rewards = rewards.view(G,group_size)
        mean = rewards.mean(dim=-1,keepdim=True)
        std = rewards.std(dim=-1,keepdim=True)
        advantages =(rewards - mean) / (std+self.eps)
        return advantages.view(B).detach()

    def compute_loss(self,new_log_probs,old_log_probs,ref_log_probs,rewards,group_size,mask):
        advantages = self.group_advantages(rewards,group_size)
        advantages = advantages.unsqueeze(-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip,1.0 + self.clip) * advantages
        log_ratio =ref_log_probs - new_log_probs
        kl_score = (torch.exp(log_ratio) - log_ratio -1) * self.beta
        per_token_loss = -torch.min(surr1, surr2) + kl_score
        masked_loss = per_token_loss * mask
        mean_loss = masked_loss.sum(dim=-1) / mask.sum(dim=-1)
        return mean_loss.mean()
