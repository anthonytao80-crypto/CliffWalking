import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gridworld as gw


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出在 -1 到 1 之间
        )

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=1))


class DDPG_Agent():
    def __init__(self, obs_dim, action_dim, lr_actor=0.001, lr_critic=0.002, gamma=0.9, tau=0.005):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # Actor 网络
        self.actor = Actor(obs_dim, action_dim)
        self.actor_target = Actor(obs_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic 网络
        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = Critic(obs_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def _one_hot(self, obs):
        vec = np.zeros((1, self.obs_dim))
        vec[0][obs] = 1.0
        return torch.FloatTensor(vec)

    def get_action(self, obs, noise=0.1):
        state = self._one_hot(obs)
        self.actor.eval()
        with torch.no_grad():
            action_cont = self.actor(state).numpy()[0]
        self.actor.train()

        # 加入噪声探索
        action_cont += noise * np.random.randn(self.action_dim)
        action_cont = np.clip(action_cont, -1, 1)

        # 离散化：取数值最大的索引作为动作
        return np.argmax(action_cont), action_cont

    def learn(self):
        if len(self.memory) < self.batch_size: return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([t[0] for t in batch])
        actions_cont = torch.FloatTensor(np.array([t[1] for t in batch]))
        rewards = torch.FloatTensor([t[2] for t in batch]).unsqueeze(1)
        next_states = torch.cat([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch]).unsqueeze(1)

        # 1. 更新 Critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, target_actions)

        current_q = self.critic(states, actions_cont)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. 更新 Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. 软更新 Target Networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# 训练管理器
class TrainManager():
    def __init__(self, env):
        self.env = env
        self.agent = DDPG_Agent(obs_dim=env.observation_space.n, action_dim=env.action_space.n)

    def train(self, episode_num=5000):
        # --- 新增：记录历史最高奖励 ---
        best_reward = -float('inf')

        for e in range(episode_num):
            obs, _ = self.env.reset()
            total_reward = 0
            while True:
                action_idx, action_cont = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action_idx)
                done = terminated or truncated

                self.agent.memory.append((
                    self.agent._one_hot(obs),
                    action_cont,
                    reward,
                    self.agent._one_hot(next_obs),
                    float(done)
                ))

                self.agent.learn()
                obs = next_obs
                total_reward += reward
                if done: break
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.agent.actor.state_dict(), "ddpg_actor_best.pth")
                print(f"Episode {e}: 新的最佳奖励 {best_reward:.2f}! 模型已更新。")

            if e % 10 == 0:
                print(f"Episode {e}: Current Reward = {total_reward:.2f}, Best So Far = {best_reward:.2f}")

        print(f"训练结束。最高奖励为: {best_reward:.2f}, 模型保存在 ddpg_actor_best.pth")


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    env = gw.FrozenLakeWapper(env)
    manager = TrainManager(env)
    manager.train()