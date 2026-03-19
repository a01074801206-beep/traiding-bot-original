import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from datetime import datetime

class MonsterTrainManager:
    def __init__(self, model, lr=1e-4, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = torch.nn.MSELoss() # 수익률 예측용 평균제곱오차
        
        # RTX 4060의 FP16 가속을 위한 Mixed Precision 설정 (학습 속도 2배 향상)
        self.scaler = torch.cuda.amp.GradScaler()

    def prepare_data(self, tensor_path, seq_len=60):
        """32GB RAM에서 5D 텐서를 불러와 학습용 시퀀스로 변환"""
        print(f"📂 데이터 로딩 중: {tensor_path}")
        df = pd.read_parquet(tensor_path)
        
        # 1. Feature 분리 (5D 텐서의 열들을 채널별로 슬라이싱)
        price_cols = ['open', 'high', 'low', 'close', 'volume'] # 실제론 더 많음
        macro_cols = ['nasdaq', 'usd_krw', 'bdi_proxy', 'wti_oil']
        nlp_cols = ['sent_avg', 'sent_std', 'hype_idx', 'max_impact', 'min_impact']
        target_col = 'target_return' # 다음날 수익률

        # 2. 시계열 슬라이싱 (i5-14400 멀티코어 연산 활용 가능)
        # 여기서는 단순화를 위해 numpy 벡터화 연산 사용
        def create_sequences(data, cols, seq_len):
            sequences = []
            for i in range(len(data) - seq_len):
                sequences.append(data[cols].iloc[i:i+seq_len].values)
            return np.array(sequences, dtype=np.float32)

        print("🔄 시계열 시퀀스 생성 중... (32GB RAM 활용)")
        # 종목별로 루프를 돌며 시퀀스를 생성해야 함 (실제 구현 시 종목별 분리 로직 추가)
        p_seq = create_sequences(df, price_cols, seq_len)
        m_seq = create_sequences(df, macro_cols, seq_len)
        n_seq = create_sequences(df, nlp_cols, seq_len)
        target = df[target_col].iloc[seq_len:].values.astype(np.float32)

        # 3. TensorDataset으로 변환
        dataset = TensorDataset(
            torch.from_numpy(p_seq), 
            torch.from_numpy(m_seq), 
            torch.from_numpy(n_seq), 
            torch.from_numpy(target)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for p, m, n, y in dataloader:
            p, m, n, y = p.to(self.device), m.to(self.device), n.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # RTX 4060 텐서 코어 가속 가동
            with torch.cuda.amp.autocast():
                output = self.model(p, m, n)
                loss = self.criterion(output.squeeze(), y)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def save_checkpoint(self, path="models/monster_best.pth"):
        torch.save(self.model.state_code(), path)
        print(f"💾 모델 저장 완료: {path}")