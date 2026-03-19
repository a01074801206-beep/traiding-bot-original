import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# 1. 신경망 구조 (LSTM + Dense)
class DQNet(nn.Module):
    def __init__(self, input_dim, feature_dim, action_dim=3):
        super(DQNet, self).__init__()
        # 시계열 데이터 처리를 위한 LSTM 레이어
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=64, num_layers=2, batch_first=True)
        # 최종 의사결정을 위한 Fully Connected 레이어
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim) # 0:관망, 1:매수, 2:매도
        )

    def forward(self, x):
        # x shape: (batch, window_size, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # 마지막 타임스텝의 출력만 사용
        return out

# 2. 강화학습 에이전트 로직
class DQNAgent:
    def __init__(self, state_dim, feature_dim, action_dim=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNet(state_dim, feature_dim, action_dim).to(self.device)
        self.target_model = DQNet(state_dim, feature_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000) # 경험 재현 메모리
        self.gamma = 0.95 # 할인율
        self.epsilon = 1.0 # 탐험률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([0, 1, 2])
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size: return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 현재 Q값과 타겟 Q값 계산
        curr_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
