import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import gridworld as gw

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 必须包含 Tanh，与训练保持一致
        )

    def forward(self, x):
        return self.fc(x)


def run_demo():
    env = gym.make('FrozenLake-v1', render_mode="human")
    #env = gw.FrozenLakeWapper(env)

    # env = gym.make('CliffWalking-v0')
    # env = gw.CliffWalkingWapper(env)

    obs_dim = env.observation_space.n
    action_dim = env.action_space.n

    model = Actor(obs_dim, action_dim)
    try:
        model.load_state_dict(torch.load("ddpg_actor.pth"))
        print("DDPG Actor 模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    model.eval()

    # 3. 运行演示
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    print("正在演示 DDPG 决策路径...")
    while not done and step_count < 100:
        # 渲染 turtle 画面
        env.render()

        # 状态预处理 (One-hot)
        state_vec = np.zeros((1, obs_dim))
        state_vec[0][obs] = 1.0
        state_tensor = torch.FloatTensor(state_vec)

        # 4. 预测动作
        with torch.no_grad():
            # Actor 输出的是连续向量，例如 [0.1, 0.9, -0.2, 0.4]
            continuous_action = model(state_tensor).numpy()[0]
            # 离散化：取数值最大的索引作为执行动作
            action = np.argmax(continuous_action)

        # 执行
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1

        # 减慢速度方便肉眼观察
        time.sleep(0.2)

    # 最后一帧渲染
    env.render()
    print(f"演示结束！总步数: {step_count}, 总奖励: {total_reward:.2f}")

    # 保持窗口一会儿
    time.sleep(2)
    env.close()


if __name__ == "__main__":
    run_demo()