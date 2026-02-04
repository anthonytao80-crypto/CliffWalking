import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gridworld as gw

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class Q_Agent():
    def __init__(self, obs_dim: int, action_dim: int, epsilon: float = 0.1, lr: float = 0.001,
                 gamma: float = 0.9) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon

        # 实例化网络和优化器
        self.model = QNet(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _one_hot(self, obs):
        """将离散状态 [5] 转换为 one-hot 向量 [[0,0,0,0,0,1,0...]]"""
        vec = np.zeros((1, self.obs_dim))
        vec[0][obs] = 1.0
        return torch.FloatTensor(vec)

    def get_target_action(self, obs: int) -> int:
        state_tensor = self._one_hot(obs)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def get_behavior_action(self, obs: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return self.get_target_action(obs)

    def learn(self, obs: int, action: int, reward: float, next_obs: int, done: bool) -> None:
        # 准备数据
        s = self._one_hot(obs)
        s_next = self._one_hot(next_obs)

        # 计算当前 Q 值
        q_values = self.model(s)
        current_q = q_values[0][action]

        # 计算目标 Q 值 (TD Target)
        with torch.no_grad():
            next_q_values = self.model(s_next)
            max_next_q = torch.max(next_q_values)
            target_q = reward + (1 - float(done)) * self.gamma * max_next_q

        # 梯度下降
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class TrainManager():
    def __init__(self, env, episode_num=2000):
        self.env = env
        self.episode_num = episode_num
        self.agent = Q_Agent(
            obs_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            lr=0.001
        )

    def train(self):
        for e in range(self.episode_num):
            obs, _ = self.env.reset()
            total_reward = 0
            while True:
                action = self.agent.get_behavior_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 执行神经网络学习
                self.agent.learn(obs, action, reward, next_obs, done)

                obs = next_obs
                total_reward += reward
                if done: break

            # 逐渐降低探索率 (Epsilon Decay)
            self.agent.epsilon = max(0.01, self.agent.epsilon * 0.998)

            if e % 20 == 0:
                print(f"Episode {e}: Reward = {total_reward}, Epsilon = {self.agent.epsilon:.3f}")
        torch.save(self.agent.model.state_dict(), "cliff_walking_model.pth")
        print("模型已保存为 cliff_walking_model.pth")

# env = gym.make('CliffWalking-v1')
# env = gw.CliffWalkingWapper(env)

env = gym.make('FrozenLake-v1')
env = gw.FrozenLakeWapper(env)

manager = TrainManager(env)
manager.train()