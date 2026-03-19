import numpy as np
import pandas as pd
from atomic_data_loader import AtomicDataLoader
from trading_strategy_agent import AssetAllocationAgent

class TradingGymEnv:
    """
    사용자 정의 2,700개 종목 원자적 데이터를 활용한 강화학습 환경.
    5만 원 시드머니 분배 전략(20:15:65)을 강제 적용함.
    """
    def __init__(self, strategy_slot='SCALPING_20', window_size=30):
        self.loader = AtomicDataLoader()
        self.agent = AssetAllocationAgent(seed_money=50000)
        
        self.strategy_slot = strategy_slot
        self.window_size = window_size
        
        # 해당 전략 슬롯에 속하는 종목들 로드
        self.tickers = self.loader.get_tickers_by_strategy(strategy_slot)
        self.current_ticker_idx = 0
        self.current_step = 0
        
        # 현재 학습 중인 종목의 전체 데이터
        self.data = None 
        self._load_next_ticker_data()

    def _load_next_ticker_data(self):
        """다음 학습 종목 데이터를 로더에서 가져옴"""
        if self.current_ticker_idx >= len(self.tickers):
            self.current_ticker_idx = 0 # 종목 순환
            
        ticker = self.tickers[self.current_ticker_idx]
        # 스캘핑/낙주/안정 전략에 필요한 지표 세트 정의
        features = ['close', 'volume', 'price_volatility', 'rsi_14', 'nasdaq']
        
        self.data = self.loader.assemble_row(ticker, features)
        self.current_ticker = ticker
        self.current_step = self.window_size
        self.current_ticker_idx += 1

    def reset(self):
        """환경 초기화 (새로운 종목 혹은 처음부터 시작)"""
        self._load_next_ticker_data()
        return self._get_observation()

    def _get_observation(self):
        """현재 시점의 window_size만큼의 텐서 반환"""
        obs = self.data.iloc[self.current_step - self.window_size : self.current_step].values
        return obs

    def step(self, action):
        """
        AI의 행동(Action)을 받아 상태 변화와 보상을 계산
        action: 0 (관망/Hold), 1 (매수/Buy), 2 (매도/Sell)
        """
        current_price = self.data.iloc[self.current_step]['close']
        done = False
        
        # 1. 에이전트를 통한 실제 거래 실행 및 보상 계산
        reward = 0
        if action == 1: # 매수
            qty = self.agent.get_betting_size(self.strategy_slot, current_price)
            if qty > 0:
                self.agent.execute_trade(self.strategy_slot, self.current_ticker, current_price, qty, 'BUY')
                reward = 0.1 # 진입 자체에 대한 소폭 보상 (탐험 유도)
            else:
                reward = -0.5 # 잔고 부족 등으로 매수 실패 시 벌점
                
        elif action == 2: # 매도
            if self.current_ticker in self.agent.slots[self.strategy_slot]['holdings']:
                hold_qty = self.agent.slots[self.strategy_slot]['holdings'][self.current_ticker]
                # 매수 시점 대비 수익률 계산 (Agent 내 로직 활용 가능)
                # 여기서는 단순화하여 매도 실행
                self.agent.execute_trade(self.strategy_slot, self.current_ticker, current_price, hold_qty, 'SELL')
                reward = 1.0 # 수익 실현에 대한 기본 보상 (실제는 수익률 비례로 상세화 필요)
            else:
                reward = -0.5 # 미보유 종목 매도 시도 벌점

        # 2. 다음 단계로 이동
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            
        next_obs = self._get_observation()
        
        return next_obs, reward, done, {"ticker": self.current_ticker, "balance": self.agent.total_balance}

if __name__ == "__main__":
    # 환경 테스트
    env = TradingGymEnv(strategy_slot='SCALPING_20')
    obs = env.reset()
    
    print(f">>> 학습 환경 준비 완료. 종목: {env.current_ticker}")
    
    # 랜덤 행동 테스트 (10 step)
    for _ in range(10):
        random_action = np.random.choice([0, 1, 2])
        next_obs, reward, done, info = env.step(random_action)
        print(f"Action: {random_action} | Reward: {reward:.2f} | Ticker: {info['ticker']}")
        if done: break
