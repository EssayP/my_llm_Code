import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128,action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128,1)

    def forward(self,x):
        x = self.base(x)
        probs = self.actor(x)
        value = self.critic(x)
        return probs,value


class PPO:
    def __init__(self,state_dim,action_dim,lr=3e-4,gamma=0.99,
                 eps_clip=0.2,K_epochs=4,gae_lambda=0.95):
        self.policy = ActorCritic(state_dim,action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=lr)
        self.policy_old = ActorCritic(state_dim,action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda


    def select_action(self,state):
        """
        用旧策略根据当前状态采样一个动作，并记录该动作的 log probability，供 PPO 更新使用
        :param state:
        :return:
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy_old(state)
            dist = Categorical(probs) #构造离散分布
            action = dist.sample() #从概率分布中采样动作
            log_prob = dist.log_prob(action) #计算logπ(a|s),用作后面计算重要性比率
        return action.item(),log_prob

    def compute_gae(self,rewards,values,dones):
        """
        compute_gae 用 TD error 递归计算优势函数，
        通过引入 λ 实现 bias-variance tradeoff，让策略梯度更稳定
        :param rewards: 每一步的奖励
        :param values: critic预测的V(s)
        :param dones: 是否终止
        :return:
        """
        advantages = []
        gae = 0
        values = values.squeeze(-1)
        #从后往前倒推 At=δt+γλA_t+1 当前误差＋未来误差的衰减累积
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                # δt=rt+γV(s_t+1)−V(st)
                delta = rewards[step] + self.gamma * values[step + 1]- values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0,gae)

        advantages = torch.tensor(advantages,dtype=torch.float32)
        return advantages

    def update(self,states,actions,log_probs,rewards,dones):
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)

        with torch.no_grad():
            _,values = self.policy(states) #获取V(s_t)

        advantages = self.compute_gae(rewards,values,dones) #获取A_t
        returns = advantages + values.squeeze().detach() #目标value: R_t = A_t + V(s_t)

        advantages = (advantages - advantages.mean()) / (advantages.std()+1e-8)

        for _ in range(self.K_epochs):
            probs,state_values = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1- self.eps_clip,1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1,surr2)

            critic_loss = nn.MSELoss()(state_values.squeeze(),returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())