import gym
import numpy as np
from src.config.trading_config import TradingConfig

class TradingGymEnv(gym.Env):
    def __init__(self):
        super(TradingGymEnv, self).__init__()
        self.df = None
        
        # 액션 공간: 0(관망), 1(매수), 2(매도)
        self.action_space = gym.spaces.Discrete(3)
        # 상태 공간: FeatureConfig의 input_dim에 맞춰 동적으로 설정됨
        # (실제 구현 시 observation_space는 reset 시점의 df 컬럼 수로 정의 가능)

    def reset(self, df_from_loader, initial_balance=50000):
        """
        로더(Loader)와 융합기(Fuser)가 완성한 데이터를 주입받습니다.
        990 Pro에서 로드된 15~50종의 데이터가 이 df에 들어있습니다.
        """
        self.df = df_from_loader
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        
        return self._get_observation()

    def _get_reward(self, current_total_assets, daily_return):
        """
        TradingConfig를 참조하여 자산 구간별 보상을 계산합니다.
        """
        # 현재 RSI (데이터 도킹된 컬럼에서 추출)
        rsi = self.df.iloc[self.current_step].get('rsi', 50) 

        # 1단계: 100만 원 미만 (공격적)
        if current_total_assets < TradingConfig.SCALPING_THRESHOLD:
            return daily_return * 100
        
        # 2단계: 100만 ~ 200만 (스캘핑 85% + 낙수 15%)
        elif current_total_assets < TradingConfig.STABLE_THRESHOLD:
            scalping_reward = daily_return * 0.85
            dip_reward = (daily_return if rsi < 30 else 0) * 0.15
            return (scalping_reward + dip_reward) * 1.5
            
        # 3단계: 200만 원 이상 (안정 65% + 스캘핑 20% + 낙수 15%)
        else:
            # 변동성 대비 수익(안정성) 계산 로직 (간략화)
            stable_reward = daily_return * 0.65 
            scalping_reward = daily_return * 0.20
            dip_reward = (daily_return if rsi < 30 else 0) * 0.15
            return stable_reward + scalping_reward + dip_reward

    def step(self, action):
        # 1. 현재 가격 정보
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['close']

        # 2. 액션 처리 (TradingConfig의 세금/수수료 적용)
        if action == 1: # BUY
            if self.balance > current_price:
                # 수수료 및 슬리피지 계산
                total_cost = current_price * (1 + TradingConfig.COMMISSION_RATE + TradingConfig.SLIPPAGE)
                self.shares_held = self.balance // total_cost
                self.balance -= (self.shares_held * total_cost)

        elif action == 2: # SELL
            if self.shares_held > 0:
                # 세금 및 수수료 계산
                tax_and_fee = TradingConfig.TAX_RATE + TradingConfig.COMMISSION_RATE
                proceeds = self.shares_held * current_price * (1 - tax_and_fee - TradingConfig.SLIPPAGE)
                self.balance += proceeds
                self.shares_held = 0

        # 3. 상태 업데이트 및 보상 계산
        self.current_step += 1
        total_assets = self.balance + (self.shares_held * current_price)
        daily_return = (total_assets - self.initial_balance) / self.initial_balance
        
        reward = self._get_reward(total_assets, daily_return)
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # 현재 시점의 모든 피처(도킹된 데이터 전체)를 벡터로 반환
        return self.df.iloc[self.current_step].values
