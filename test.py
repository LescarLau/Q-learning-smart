import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)  # 输入状态大小为 state_size，输出大小为 hidden_size 的全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 隐藏层全连接层，输入输出大小都为 hidden_size
        self.fc3 = nn.Linear(hidden_size, action_size)  # 输出动作大小的全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数进行非线性变换
        x = torch.relu(self.fc2(x))  # 同上
        x = self.fc3(x)  # 输出层，无激活函数
        return x

# 定义 Double DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, epsilon, epsilon_decay):
        self.state_size = state_size  # 状态大小
        self.action_size = action_size  # 动作大小
        self.lr = lr  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的 CUDA 设备

        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)  # 创建 Q 网络
        self.target_q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)  # 创建目标 Q 网络
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # 将目标 Q 网络初始化为与 Q 网络相同的权重
        self.target_q_network.eval()  # 将目标 Q 网络设置为评估模式（不更新权重）

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)  # 使用 Adam 优化器进行优化
        self.loss_fn = nn.MSELoss()  # 定义均方误差损失函数

    def select_action(self, state):
        if random.random() < self.epsilon:  # 如果随机数小于探索率
            return random.randint(0, self.action_size - 1)  # 随机选择一个动作
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 转换状态为张量并添加批次维度
                q_values = self.q_network(state_tensor)  # 获取 Q 值
                return q_values.argmax().item()  # 返回 Q 值最大的动作

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:  # 如果回放缓冲区中的样本数量小于批次大小
            return

        samples = random.sample(replay_buffer, batch_size)  # 从回放缓冲区中随机采样一个批次的样本
        states, actions, rewards, next_states, dones = zip(*samples)  # 拆分样本为状态、动作、奖励、下一个状态、完成标志

        states = torch.FloatTensor(states).to(self.device)  # 转换状态为张量
        actions = torch.LongTensor(actions).to(self.device)  # 转换动作为张量
        rewards = torch.FloatTensor(rewards).to(self.device)  # 转换奖励为张量
        next_states = torch.FloatTensor(next_states).to(self.device)  # 转换下一个状态为张量
        dones = torch.FloatTensor(dones).to(self.device)  # 转换完成标志为张量

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))  # 计算当前状态动作值
        next_actions = self.q_network(next_states).max(1)[1]  # 选择下一个状态下 Q 值最大的动作
        next_q_values = self.target_q_network(next_states).gather(1, next_actions.unsqueeze(1)).detach()  # 获取目标 Q 值
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values  # 计算目标 Q 值

        loss = self.loss_fn(current_q_values, target_q_values)  # 计算损失

        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新网络权重

        self.epsilon *= self.epsilon_decay  # 更新探索率

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # 更新目标 Q 网络的权重为 Q 网络的权重

# 环境和超参数
env_name = 'CartPole-v1'
state_size = 4  # CartPole-v1环境中的状态大小
action_size = 2  # CartPole-v1环境中的动作大小
hidden_size = 64  # 隐藏层大小
lr = 0.001  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.995  # 探索率衰减率
batch_size = 64  # 批次大小
buffer_size = 10000  # 经验回放缓冲区大小
update_target_freq = 100  # 更新目标 Q 网络的频率

# 初始化环境
env = gym.make(env_name)
replay_buffer = []

# 初始化智能体
agent = DoubleDQNAgent(state_size, action_size, hidden_size, lr, gamma, epsilon, epsilon_decay)

# 模拟训练过程
for episode in range(1000):
    current_state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(current_state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((current_state, action, reward, next_state, done))
        current_state = next_state
        total_reward += reward

        agent.train(replay_buffer, batch_size)

        if len(replay_buffer) > buffer_size:
            replay_buffer.pop(0)

    if episode % update_target_freq == 0:
        agent.update_target_network()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

env.close()
