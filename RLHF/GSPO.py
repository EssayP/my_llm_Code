import torch
import torch.nn as nn
import torch.nn.functional as F
class GSPOLoss:
    def __init__(self,clip,beta,eps):
        self.beta = beta
        self.clip = clip
        self.eps = eps

    def group_advantages(self,rewards,group_size):
        B = rewards.shape[0]
        G = B // group_size
        rewards=rewards.view(G,group_size)
        mean = rewards.mean(dim=-1,keepdim=True)
        std = rewards.std(dim=-1,keepdim=True)
        advantages = (rewards - mean) /(std+self.eps)
        return advantages.view(B).detach()

    def compute_loss(self,new_logprobs,old_logprobs,ref_logprobs,rewards,group_size,mask):
        advantages = self.group_advantages(rewards,group_size)
        advantages = advantages.unsqueeze(-1)

        per_token_log_ratio =new_logprobs - old_logprobs
        seq_log_ratio = (per_token_log_ratio * mask).sum(dim=-1,keepdim=True)
        seq_ratio = torch.exp(seq_log_ratio)

        surr1 = seq_ratio * advantages
        surr2 = torch.clamp(seq_ratio,1-self.clip,1+self.clip) * advantages
        policy_loss = -torch.min(surr1,surr2)

        log_ratio = ref_logprobs - new_logprobs
        per_token_kl= torch.exp(log_ratio) - log_ratio - 1
        kl_score = self.beta * (per_token_kl * mask).sum(dim=-1,keepdim=True) / (mask.sum(dim=-1,keepdim=True) + self.eps)

        per_seq_loss = policy_loss + kl_score
        loss = per_seq_loss.mean()
        return loss


    
