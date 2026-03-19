import os
import torch
from src.utils.paths import CHECKPOINT_DIR

class DQNAgent:
    def save(self, filename):
        # 경로를 CHECKPOINT_DIR로 강제 지정
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), save_path)
