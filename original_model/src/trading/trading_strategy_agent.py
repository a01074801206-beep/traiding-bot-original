import yaml
import os

class AssetAllocationAgent:
    def __init__(self, config_path="../../config.yaml", current_seed=50000):
        # 1. 설정 로드
        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))
        with open(full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.total_balance = current_seed
        self.update_allocation_by_balance()

    def update_allocation_by_balance(self):
        """현재 잔고에 맞는 전략 비중을 찾아 슬롯 재배정"""
        phases = self.config['strategy']['phases']
        selected_phase = None
        
        for phase in phases:
            if phase['min_balance'] <= self.total_balance < phase['max_balance']:
                selected_phase = phase
                break
        
        if not selected_phase:
            selected_phase = phases[-1] # 예외 시 마지막 단계 적용

        # 2. 슬롯별 자금 재분배
        self.slots = {
            'SCALPING': {
                'ratio': selected_phase['scalping'],
                'current_cash': self.total_balance * selected_phase['scalping'],
                'holdings': {}
            },
            'CONTRARIAN': {
                'ratio': selected_phase['contrarian'],
                'current_cash': self.total_balance * selected_phase['contrarian'],
                'holdings': {}
            },
            'STABLE': {
                'ratio': selected_phase['stable'],
                'current_cash': self.total_balance * selected_phase['stable'],
                'holdings': {}
            }
        }
        print(f"🔄 전략 업데이트 완료: 잔고 {self.total_balance:,.0f}원 기준")
        print(f"   [스캘핑: {selected_phase['scalping']*100:.0f}% | 낙주: {selected_phase['contrarian']*100:.0f}% | 안정: {selected_phase['stable']*100:.0f}%]")

    def execute_trade_success(self, profit_or_loss):
        """매매 종료 후 잔고가 바뀌면 비중 자동 재계산"""
        self.total_balance += profit_or_loss
        self.update_allocation_by_balance()
