import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 定义状态空间和动作空间
states = [(256, 5), (512, 10)]  # 函数状态：输入长度和迭代次数
actions = [794, 1000]  # 可选的资源配置

# 定义神经网络模型
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(actions))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义DQN Agent
class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = 0.1

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.model(state)
                return actions[torch.argmax(q_values).item()]

    def remember(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        for _ in range(num_episodes):
            state = random.choice(states)
            while state != (512, 10):
                action = self.select_action(state)
                next_state = (512, 10) if action == 1000 else (256, 5)
                reward_val = 1 if action == actions[states.index((256, 5))] else 0
                self.remember(state, action, next_state, reward_val)
                state = next_state
                self.replay()

# 训练 DQN 智能体
agent = DQNAgent()
agent.train(1000)

# 输出学习到的最佳资源配置
best_actions = []
for state in states:
    action = agent.select_action(state)
    best_actions.append(action)

print("Learned best actions:", best_actions)
