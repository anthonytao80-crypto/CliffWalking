#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import gymnasium as gym
import turtle
import numpy as np
import random

# turtle tutorial : https://docs.python.org/3.3/library/turtle.html

def GridWorld(gridmap=None, is_slippery=False):
    if gridmap is None:
        gridmap = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
    env = gym.make("FrozenLake-v1", desc=gridmap, is_slippery=False)
    env = FrozenLakeWapper(env)
    return env


class FrozenLakeWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.max_y = env.unwrapped.desc.shape[0]
        self.max_x = env.unwrapped.desc.shape[1]
        self.t = None
        self.unit = 50
        self.s = 0
        for r in range(self.max_y):
            for c in range(self.max_x):
                if env.unwrapped.desc[r][c] == b'G':
                    self.goal_pos = (c, self.max_y - 1 - r)  # 转换为坐标系

    def get_pos(self, s):
        """将状态索引转为 (x, y) 坐标"""
        x = s % self.max_x
        y = self.max_y - 1 - int(s / self.max_x)
        return x, y

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for _ in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for i in range(self.unwrapped.desc.shape[0]):
                for j in range(self.unwrapped.desc.shape[1]):
                    x = j
                    y = self.max_y - 1 - i
                    if self.unwrapped.desc[i][j] == b'S':  # Start
                        self.draw_box(x, y, 'white')
                    elif self.unwrapped.desc[i][j] == b'F':  # Frozen ice
                        self.draw_box(x, y, 'white')
                    elif self.unwrapped.desc[i][j] == b'G':  # Goal
                        self.draw_box(x, y, 'yellow')
                    elif self.unwrapped.desc[i][j] == b'H':  # Hole
                        self.draw_box(x, y, 'black')
                    else:
                        self.draw_box(x, y, 'white')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.s = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.s = obs

        curr_x, curr_y = self.get_pos(obs)
        goal_x, goal_y = self.goal_pos

        # 计算曼哈顿距离
        dist = abs(curr_x - goal_x) + abs(curr_y - goal_y)

        if terminated:
            if reward == 1.0:
                reward = 10.0  # 终点大奖
            else:
                reward = -10.0  # 掉坑惩罚
        else:
            # 核心改进：距离奖励（离得越近，值越大，通常是负向引导）
            # 比如：-0.1 (每步惩罚) - 0.1 * 距离
            reward = -0.1 - 0.1 * dist

        return obs, reward, terminated, truncated, info

class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4
        self.s = 0

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)


def get_random_env():
    """随机返回一个包装好的环境实例"""
    choice = random.randint(0, 2)

    if choice == 0:
        # 环境 1: 标准悬崖行走
        env = gym.make("CliffWalking-v1")
        return CliffWalkingWapper(env)

    elif choice == 1:
        # 环境 2: FrozenLake (简单版)
        return GridWorld(['SFFF', 'FHFH', 'FFFH', 'HFFG'])

    else:
        # 环境 3: 复杂的自定义地图
        custom_map = [
            'SFFFF',
            'FHHHH',
            'FFFFF',
            'HHFHG'
        ]
        return GridWorld(custom_map)

if __name__ == '__main__':
    # 环境1：FrozenLake, 可以配置冰面是否是滑的
    # 0 left, 1 down, 2 right, 3 up
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = FrozenLakeWapper(env)

    # 环境2：CliffWalking, 悬崖环境
    # env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    # env = CliffWalkingWapper(env)

    # 环境3：自定义格子世界，可以配置地图, S为出发点Start, F为平地Floor, H为洞Hole, G为出口目标Goal
    # gridmap = [
    #         'SFFF',
    #         'FHFF',
    #         'FFFF',
    #         'HFGF' ]
    # env = GridWorld(gridmap)

    env.reset()
    for step in range(10):
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(\
                step, action, obs, reward, done, info))
        env.render()  # 渲染一帧图像