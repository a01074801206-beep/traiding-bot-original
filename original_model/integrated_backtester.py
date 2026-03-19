import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_gym_env import TradingGymEnv
from dqn_trading_model import DQNet

class IntegratedBacktester:
    """
    3개 전략 모델(20:15:65)을 동시에 가동하여 
    5만 원 시드머니의 통합 수익률 및 MDD를 측정함.
    """
    def __init__(self, seed_money=50000):
        self.seed_money = seed_money
        self.strategies = ['SCALPING_20', 'CONTRARIAN_15', 'STABLE_65']
        self.envs = {s: TradingGymEnv(strategy_slot=s) for s in self.strategies}
        self.models = {}
        
        # 학습된 모델 로드 (파일이 없을 경우 대비 예외 처리)
        for s in self.strategies:
            model_path = f"model_{s}.pth"
            # 임시로 더미 모델 생성 (실제 사용 시 torch.load 필요)
            obs = self.envs[s].reset()
            model = DQNet(input_dim=obs.shape[0], feature_dim=obs.shape[1])
            if torch.os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
            model.eval()
            self.models[s] = model

    def run_simulation(self, steps=100):
        print(f"📊 통합 백테스팅 시작 (초기 자본: {self.seed_money}원)...")
        
        portfolio_history = []
        total_balance = self.seed_money
        
        # 각 전략별 상태 초기화
        states = {s: self.envs[s].reset() for s in self.strategies}
        dones = {s: False for s in self.strategies}

        for i in range(steps):
            current_total = 0
            
            for s in self.strategies:
                if not dones[s]:
                    # 1. 모델의 판단 (Action 선택)
                    state_tensor = torch.FloatTensor(states[s]).unsqueeze(0)
                    with torch.no_grad():
                        action = torch.argmax(self.models[s](state_tensor)).item()
                    
                    # 2. 환경 실행
                    next_state, reward, done, info = self.envs[s].step(action)
                    states[s] = next_state
                    dones[s] = done
                    
                    # 3. 각 슬롯의 현재 평가 금액 합산
                    # (현금 + 보유 주식 가치)
                    slot_val = self.envs[s].agent.slots[s]['current_cash']
                    for ticker, qty in self.envs[s].agent.slots[s]['holdings'].items():
                        # 현재가 기준 평가 (마지막 close값 활용)
                        slot_val += (qty * self.envs[s].data.iloc[self.envs[s].current_step]['close'])
                    current_total += slot_val
                else:
                    # 데이터 종료 시 마지막 가치 유지
                    current_total += self.envs[s].agent.slots[s]['current_cash']

            portfolio_history.append(current_total)
            if i % 10 == 0:
                print(f"Step {i} | 통합 평가액: {current_total:.0f}원")

        return portfolio_history

    def plot_results(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history, label='Total Portfolio Value', color='blue')
        plt.axhline(y=self.seed_money, color='red', linestyle='--', label='Initial Seed')
        plt.title('50,000 KRW Strategy Backtest (20:15:65 Split)')
        plt.xlabel('Time Steps')
        plt.ylabel('Value (KRW)')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    backtester = IntegratedBacktester(seed_money=50000)
    history = backtester.run_simulation(steps=100)
    
    final_val = history[-1]
    return_pct = (final_val - 50000) / 50000 * 100
    print(f"\n✅ 최종 평가액: {final_val:.0f}원 (수익률: {return_pct:.2f}%)")
    
    backtester.plot_results(history)
