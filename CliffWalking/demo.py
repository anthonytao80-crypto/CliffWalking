import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
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


def run_demo():
    # env = gym.make('CliffWalking-v1', render_mode="human")
    # env = gw.CliffWalkingWapper(env)
    env = gym.make('FrozenLake-v1', render_mode="human")
    env = gw.FrozenLakeWapper(env)
    obs_dim = env.observation_space.n
    action_dim = env.action_space.n

    #加载模型
    model = QNet(obs_dim, action_dim)
    model.load_state_dict(torch.load("cliff_walking_model.pth"))
    model.eval()  # 切换到评价模式
    print("模型加载成功！开始展示...")

    #运行游戏
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # 预处理状态 (One-hot)
        state_vec = np.zeros((1, obs_dim))
        state_vec[0][obs] = 1.0
        state_tensor = torch.FloatTensor(state_vec)

        # 预测动作 (纯贪婪策略，不再探索)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        # 执行动作
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # 稍微停顿一下，方便肉眼观察
        time.sleep(0.2)

    print(f"演示结束，总奖励: {total_reward}")
    env.close()


if __name__ == "__main__":
    run_demo()