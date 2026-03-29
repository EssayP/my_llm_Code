import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared = self.shared_layer(state)
        action_probs = self.actor(shared)
        state_value = self.critic(shared)
        return action_probs, state_value


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.is_terminals.clear()


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=0.002,
                 gamma=0.99,
                 eps_clip=0.2,
                 K_epochs=4,
                 gae_lambda=0.95,  # 新增：GAE 的 λ 参数
                 entropy_coef=0.01,
                 vf_coef=0.5):

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]

        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        memory.states.append(state.squeeze(0))
        memory.actions.append(action.squeeze(0))
        memory.logprobs.append(log_prob.squeeze(0))

        return action.item()

    def compute_gae(self, rewards, values, is_terminals):
        """计算 Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        values = values.squeeze()  # [batch_size]

        # 从后往前计算
        for step in reversed(range(len(rewards))):
            if is_terminals[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * values[step + 1] - values[step]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        return advantages

    def update(self, memory):
        # 转为 tensor
        old_states = torch.stack(memory.states)  # [batch, state_dim]
        old_actions = torch.stack(memory.actions)  # [batch]
        old_logprobs = torch.stack(memory.logprobs)  # [batch]

        # 获取当前策略的状态值（用于 GAE）
        with torch.no_grad():
            _, state_values = self.policy(old_states)  # [batch, 1]

        # ====================== 计算 GAE ======================
        rewards = torch.tensor(memory.rewards, dtype=torch.float32)
        is_terminals = torch.tensor(memory.is_terminals, dtype=torch.bool)

        advantages = self.compute_gae(rewards, state_values, is_terminals)

        # 计算 returns = advantages + values（用于 critic loss）
        returns = advantages + state_values.squeeze().detach()

        # 归一化 advantages（强烈推荐）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ====================== PPO 更新 ======================
        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            # 计算 ratio
            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss（使用 returns）
            critic_loss = self.MseLoss(state_values.squeeze(), returns)

            # 总损失
            loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

            # 反向传播 + 梯度裁剪
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空 memory
        memory.clear()