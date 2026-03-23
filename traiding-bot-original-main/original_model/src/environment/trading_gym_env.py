import gym
import numpy as np
from src.config.trading_config import TradingConfig

class TradingGymEnv(gym.Env):
    def __init__(self):
        super(TradingGymEnv, self).__init__()
        self.df = None
        self.action_space = gym.spaces.Discrete(3) # 0:관망, 1:매수, 2:매도

    def reset(self, df_from_loader, initial_balance=50000):
        self.df = df_from_loader
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.cost_basis = 0  # 평단가 (FinRL 참고: 수익률 계산용)
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        """
        [개선사항] 시장 데이터 + 계좌 상태를 결합한 하이브리드 벡터 반환
        """
        # 1. 시장 데이터 (OHLCV + 도킹된 지표들)
        market_state = self.df.iloc[self.current_step].values
        
        # 2. 계좌 상태 (FinRL 정밀 모델링 참고)
        current_price = self.df.iloc[self.current_step]['close']
        unrealized_profit = (current_price - self.cost_basis) / (self.cost_basis + 1e-8) if self.shares_held > 0 else 0
        
        portfolio_state = np.array([
            self.balance / TradingConfig.STABLE_THRESHOLD, # 현재 잔고의 목표 달성률 (0~1 사이 정규화)
            self.shares_held * current_price / (self.balance + self.shares_held * current_price + 1e-8), # 자산 대비 주식 비중
            unrealized_profit, # 현재 보유 종목의 수익률 상태
            self.current_step / len(self.df) # 에피소드 진행률 (시간 개념 주입)
        ])
        
        # 3. 결합 (Market + Portfolio)
        return np.concatenate([market_state, portfolio_state]).astype(np.float32)

    def _get_reward(self, total_assets, daily_return):
        """
        [개선사항] FinRL의 샤프 지수 개념을 3단계 전략에 녹여냄
        """
        rsi = self.df.iloc[self.current_step].get('rsi', 50)
        
        # 1단계: 100만 원 미만 (공격적 수익 중심)
        if total_assets < TradingConfig.SCALPING_THRESHOLD:
            return daily_return * 100.0
        
        # 2단계: 100만 ~ 200만 (수수료 극복 + 낙폭 과대 매수 보너스)
        elif total_assets < TradingConfig.STABLE_THRESHOLD:
            bonus = 0.2 if rsi < 30 else 0 # 과매도 구간 매수 유도
            return (daily_return + bonus) * 2.0
            
        # 3단계: 200만 원 이상 (안정성 중심 - MDD 방어 보상)
        else:
            # 변동성 대비 수익(안정성)을 보상으로 환산
            return daily_return * 1.5

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # 매수 로직 (평단가 계산 추가)
        if action == 1: 
            if self.balance > current_price:
                buy_cost = current_price * (1 + TradingConfig.COMMISSION_RATE + TradingConfig.SLIPPAGE)
                new_shares = self.balance // buy_cost
                # FinRL 방식: 평단가 업데이트 (가중 평균)
                total_shares = self.shares_held + new_shares
                self.cost_basis = ((self.shares_held * self.cost_basis) + (new_shares * buy_cost)) / total_shares
                self.shares_held = total_shares
                self.balance -= (new_shares * buy_cost)

        # 매도 로직
        elif action == 2:
            if self.shares_held > 0:
                sell_price = current_price * (1 - TradingConfig.TAX_RATE - TradingConfig.COMMISSION_RATE - TradingConfig.SLIPPAGE)
                self.balance += (self.shares_held * sell_price)
                self.shares_held = 0
                self.cost_basis = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        total_assets = self.balance + (self.shares_held * current_price)
        daily_return = (total_assets - self.initial_balance) / self.initial_balance
        
        return self._get_observation(), self._get_reward(total_assets, daily_return), done, {}
