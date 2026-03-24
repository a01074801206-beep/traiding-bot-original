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
        self.cost_basis = 0
        self.current_step = 0
        
        # [추가] 데이터 전처리: 한글 컬럼명을 영어로 매핑하거나 직접 참조 방어
        self.price_col = '종가' if '종가' in self.df.columns else 'close'
        self.rsi_col = 'RSI' if 'RSI' in self.df.columns else 'rsi'
        
        return self._get_observation()

    def _get_observation(self):
        # 1. 시장 데이터 (OHLCV + 도킹된 지표들)
        market_state = self.df.iloc[self.current_step].values
        
        # 2. 계좌 상태 (한글 컬럼 대응)
        current_price = self.df.iloc[self.current_step][self.price_col]
        
        # 보유 종목 수익률 계산 (평단가 기준)
        unrealized_profit = (current_price - self.cost_basis) / (self.cost_basis + 1e-8) if self.shares_held > 0 else 0
        
        portfolio_state = np.array([
            self.balance / 2000000.0, # TradingConfig.STABLE_THRESHOLD 대신 가시적인 값으로 정규화
            self.shares_held * current_price / (self.balance + self.shares_held * current_price + 1e-8),
            unrealized_profit,
            self.current_step / len(self.df)
        ])
        
        # 3. 결합 및 타입 변환 (float32로 고정해야 GPU 연산 시 에러가 안 남)
        obs = np.concatenate([market_state, portfolio_state]).astype(np.float32)
        return obs

    def step(self, action):
        # 현재가 참조
        current_price = self.df.iloc[self.current_step][self.price_col]
        
        # 매수 로직 (수수료 및 슬리피지 반영)
        if action == 1: 
            if self.balance > current_price:
                # 수수료/슬리피지 계산 (TradingConfig 값 활용)
                fee_rate = getattr(TradingConfig, 'COMMISSION_RATE', 0.00015)
                slippage = getattr(TradingConfig, 'SLIPPAGE', 0.001)
                
                buy_unit_price = current_price * (1 + fee_rate + slippage)
                new_shares = self.balance // buy_unit_price
                
                if new_shares > 0:
                    total_shares = self.shares_held + new_shares
                    self.cost_basis = ((self.shares_held * self.cost_basis) + (new_shares * buy_unit_price)) / total_shares
                    self.shares_held = total_shares
                    self.balance -= (new_shares * buy_unit_price)

        # 매도 로직
        elif action == 2:
            if self.shares_held > 0:
                tax_rate = getattr(TradingConfig, 'TAX_RATE', 0.002)
                fee_rate = getattr(TradingConfig, 'COMMISSION_RATE', 0.00015)
                
                sell_price = current_price * (1 - tax_rate - fee_rate)
                self.balance += (self.shares_held * sell_price)
                self.shares_held = 0
                self.cost_basis = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 총 자산 및 보상 계산
        total_assets = self.balance + (self.shares_held * current_price)
        daily_return = (total_assets - self.initial_balance) / self.initial_balance
        
        reward = self._get_reward(total_assets, daily_return)
        
        return self._get_observation(), reward, done, {}

    def _get_reward(self, total_assets, daily_return):
        # RSI 지표 확인
        rsi = self.df.iloc[self.current_step].get(self.rsi_col, 50)
        
        # 사용자님의 단계별 보상 로직 적용
        if total_assets < 1000000: # 공격적
            return daily_return * 100.0
        elif total_assets < 2000000: # 중립
            bonus = 0.2 if rsi < 30 else 0
            return (daily_return + bonus) * 2.0
        else: # 안정 (MDD 방어)
            return daily_return * 1.5
