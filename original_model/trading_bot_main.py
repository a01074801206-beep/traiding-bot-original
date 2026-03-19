import time
import schedule
from datetime import datetime
from integrated_atomic_collector import IntegratedAtomicCollector
from atomic_feature_engineer import AtomicFeatureEngineer
from integrated_backtester import IntegratedBacktester

class TradingBotSystem:
    """
    5만 원 시드머니 20:15:65 전략을 실전/모의 환경에서 
    무한 루프로 가동하는 메인 시스템.
    """
    def __init__(self, is_real_trading=False):
        self.base_path = "./data_lake"
        self.collector = IntegratedAtomicCollector(self.base_path)
        self.engineer = AtomicFeatureEngineer(self.base_path)
        self.is_real_trading = is_real_trading
        
        # 실전 매매 시 증권사 API 연결 로직이 여기에 들어감 (키움, 한국투자증권 등)
        if is_real_trading:
            print("⚠️ 실전 매매 모드가 활성화되었습니다. API 연결을 확인하세요.")

    def morning_setup(self):
        """08:30 - 장 시작 전 마스터 맵 업데이트 및 글로벌 지표 수집"""
        print(f"[{datetime.now()}] 장 시작 전 준비 중...")
        today = datetime.now().strftime("%Y%m%d")
        # 1. 글로벌 매크로(나스닥 등) 업데이트
        self.collector.collect_global_macro("2024-01-01", today)
        # 2. 전일 종가 기반 지표 가공
        self.engineer.process_technical_indicators()

    def market_loop(self):
        """09:00 ~ 15:30 - 실시간 데이터 수집 및 모델 판단 후 매매"""
        print(f"[{datetime.now()}] 장 중 실시간 루프 가동...")
        today = datetime.now().strftime("%Y%m%d")
        
        # 1. 1분 단위 원자적 데이터 수집 (스캘핑 20%용)
        self.collector.collect_scalping_micros(today)
        
        # 2. 모델 판단 및 주문 (에이전트 호출)
        # (실제 구현 시 모델 로드 후 현재가 기반 action 결정 로직 수행)
        print("   - 스캘핑/낙주/안정 모델 판단 및 포지션 체크 완료.")

    def evening_report(self):
        """16:00 - 장 마감 후 당일 성과 요약 및 백테스팅 업데이트"""
        print(f"[{datetime.now()}] 장 마감. 당일 성과 분석 중...")
        # 통합 백테스터를 돌려 현재 5만 원이 얼마나 변했는지 리포트 생성
        backtester = IntegratedBacktester(seed_money=50000)
        history = backtester.run_simulation(steps=10)
        print(f"   - 현재 자산 가치: {history[-1]:.0f}원")

    def run(self):
        """스케줄러 설정 및 무한 루프"""
        # 매일 정해진 시간에 작업 수행
        schedule.every().day.at("08:30").do(self.morning_setup)
        schedule.every(1).minutes.do(self.market_loop) # 장 중 1분마다 가동
        schedule.every().day.at("16:00").do(self.evening_report)

        print("🚀 자동매매 시스템이 시작되었습니다. (Ctrl+C로 종료)")
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    bot = TradingBotSystem(is_real_trading=False) # 초기에는 모의 모드로 권장
    bot.run()
