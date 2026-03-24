import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.paths import CHECKPOINT_DIR

# 실제 신경망 구조 (RTX 4060에서 계산됨)
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size) # 매수, 매도, 홀딩(3가지 액션)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, action_size=3):
        # RTX 4060을 사용하기 위한 장치 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.action_size = action_size
        
        # 모델 생성 및 GPU로 전송
        self.model = QNetwork(input_dim, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"🚀 장치 설정 완료: {self.device} (RTX 4060 가동 준비)")

    def save(self, filename):
        # 경로를 CHECKPOINT_DIR로 강제 지정
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
            
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"💾 모델 저장 완료: {save_path}")

    def load(self, filename):
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
            print(f"📂 모델 로드 완료: {save_path}")
