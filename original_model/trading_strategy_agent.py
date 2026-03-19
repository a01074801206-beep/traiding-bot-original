import os
import pandas as pd
import numpy as np

class AssetAllocationAgent:
    """
    5만 원 시드머니를 3개 전략 슬롯(20:15:65)으로 배분하고
    각 슬롯의 수익률과 리스크를 독립적으로 관리하는 에이전트.
    """
    def __init__(self, seed_money=50000):
        # 1. 자금 배분 설정
        self.total_balance = seed_money
        self.slots = {
            'SCALPING_20': {'ratio': 0.20, 'budget': seed_money * 0.20, 'current_cash': seed_money * 0.20, 'holdings': {}},
            'CONTRARIAN_15': {'ratio': 0.15, 'budget': seed_money * 0.15, 'current_cash': seed_money * 0.15, 'holdings': {}},
            'STABLE_65': {'ratio': 0.65, 'budget': seed_money * 0.65, 'current_cash': seed_money * 0.65, 'holdings': {}}
        }
        
        # 2. 성과 추적용 로그
        self.history = []

    def get_betting_size(self, strategy_slot, ticker_price):
        """특정 슬롯에서 한 종목에 태울 수 있는 최대 수량 계산"""
        slot = self.slots[strategy_slot]
        if slot['current_cash'] < ticker_price:
            return 0
        
        # 슬롯 예산의 10%를 한 종목에 배정 (분산 투자 원칙)
        max_bet = slot['budget'] * 0.1
        can_buy_qty = int(min(max_bet, slot['current_cash']) // ticker_price)
        return can_buy_qty

    def execute_trade(self, strategy_slot, ticker, price, qty, side='BUY'):
        """매수/매도 실행 및 슬롯별 잔고 업데이트"""
        slot = self.slots[strategy_slot]
        trade_amount = price * qty
        
        if side == 'BUY':
            if slot['current_cash'] >= trade_amount:
                slot['current_cash'] -= trade_amount
                slot['holdings'][ticker] = slot['holdings'].get(ticker, 0) + qty
                print(f"✅ [{strategy_slot}] 매수: {ticker} | {qty}주 @ {price}원")
        
        elif side == 'SELL':
            if ticker in slot['holdings'] and slot['holdings'][ticker] >= qty:
                slot['current_cash'] += trade_amount
                slot['holdings'][ticker] -= qty
                if slot['holdings'][ticker] == 0:
                    del slot['holdings'][ticker]
                print(f"💰 [{strategy_slot}] 매도: {ticker} | {qty}주 @ {price}원")

    def calculate_reward(self, strategy_slot, entry_price, current_price):
        """강화학습을 위한 전략별 차별화된 보상(Reward) 함수"""
        profit_pct = (current_price - entry_price) / entry_price * 100
        
        # 전략별 목표에 따른 가중치 부여
        if strategy_slot == 'SCALPING_20':
            # 스캘핑은 짧은 시간에 확실한 익절(+)이 중요
            return profit_pct * 1.5 if profit_pct > 0.5 else -2.0
        
        elif strategy_slot == 'CONTRARIAN_15':
            # 낙주 매매는 '떨어지는 칼날'을 잡으므로 손절폭을 크게 주되 반등 시 보상 극대화
            return profit_pct * 2.0 if profit_pct > 3.0 else profit_pct
            
        elif strategy_slot == 'STABLE_65':
            # 안정 수익은 MDD(최대 낙폭) 관리가 중요
            return profit_pct if profit_pct > -1.0 else -5.0

    def get_summary(self):
        """현재 전체 계좌 현황 출력"""
        print("\n" + "="*40)
        print(f"💵 전체 잔고 요약 (초기 자본: 50,000원)")
        for name, data in self.slots.items():
            print(f"- {name}: 현금 {data['current_cash']:.0f}원 | 보유 {len(data['holdings'])}종목")
        print("="*40)

if __name__ == "__main__":
    # 에이전트 초기화
    agent = AssetAllocationAgent(seed_money=50000)
    
    # 예시 시나리오: 삼성전자 스캘핑 매수 시도
    samsung_price = 75000 # 실제 가격은 수집기에서 가져와야 함
    qty = agent.get_betting_size('SCALPING_20', samsung_price)
    
    if qty > 0:
        agent.execute_trade('SCALPING_20', '005930', samsung_price, qty, 'BUY')
    else:
        print("❌ 잔고 부족 혹은 단가 미달로 매수 불가")
        
    agent.get_summary()
