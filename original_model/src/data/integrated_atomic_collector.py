import os
import pandas as pd
from pykrx import stock
import yfinance as yf
from datetime import datetime
import time

class IntegratedAtomicCollector:
    """
    기존 4개 수집기를 하나로 통합.
    모든 데이터(41개 지표 대응)를 종목별/지표별로 원자적 저장.
    """
    def __init__(self, base_path="./data_lake"):
        self.base_path = base_path
        self.master_path = os.path.join(base_path, "MASTER", "ticker_master.parquet")
        
        if not os.path.exists(self.master_path):
            raise Exception("❗ [오류] 마스터 맵이 없습니다. master_mapper.py를 먼저 실행하세요.")
        
        self.master_df = pd.read_parquet(self.master_path)
        self.all_tickers = self.master_df['ticker'].tolist()
        # 스캘핑 전용 종목 리스트 (20% 슬롯)
        self.scalping_tickers = self.master_df[self.master_df['strategy_slot'] == 'SCALPING_20']['ticker'].tolist()

    def _save_atomic(self, ticker, feature_name, data):
        """데이터를 해당 종목 폴더에 개별 지표 파일로 저장"""
        if data is None or data.empty: return
        
        target_dir = os.path.join(self.base_path, ticker)
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, f"{feature_name}.parquet")
        
        # 병합 저장 (중복 제거)
        if os.path.exists(file_path):
            old_df = pd.read_parquet(file_path)
            data = pd.concat([old_df, data]).drop_duplicates()
        
        data.to_parquet(file_path)

    def collect_market_basic(self, start_date, end_date):
        """기존 기초 수집기 역할: OHLCV 및 기본 수급"""
        print(f">>> 기본 시장 데이터 수집 시작 ({start_date} ~ {end_date})...")
        for ticker in self.all_tickers:
            try:
                # 1. 가격 데이터 (Z축 기본)
                df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
                for col, eng in {'시가':'open', '고가':'high', '저가':'low', '종가':'close', '거래량':'volume'}.items():
                    self._save_atomic(ticker, eng, df[[col]])

                # 2. 투자자별 순매수 (낙주/안정 전략용)
                df_inv = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, ticker)
                self._save_atomic(ticker, "investor_net", df_inv)
                
                print(f"   [OK] {ticker} 기본 데이터 완료")
                time.sleep(0.05)
            except Exception as e:
                print(f"   [Error] {ticker}: {e}")

    def collect_scalping_micros(self, target_date):
        """스캘핑 전용: 체결강도 및 호가 데이터 (20% 슬롯 대상)"""
        print(f">>> 스캘핑 마이크로 지표 수집 ({target_date})...")
        for ticker in self.scalping_tickers:
            try:
                # 호가/체결 데이터 (실제는 틱 데이터를 1분봉 가공해야함)
                df_hoga = stock.get_market_cap_by_date(target_date, target_date, ticker)
                self._save_atomic(ticker, "market_cap_micro", df_hoga)
                # 여기에 나중에 키움 API 등 실시간 틱 가공 로직 추가
                print(f"   [Scalping OK] {ticker}")
            except Exception as e:
                continue

    def collect_global_macro(self, start_date, end_date):
        """기존 매크로 수집기 역할: 나스닥, 환율 등"""
        print(f">>> 글로벌 매크로 수집 시작...")
        macros = {'nasdaq': '^IXIC', 'semicon': '^SOX', 'usd_krw': 'USDKRW=X'}
        for name, symbol in macros.items():
            df = yf.download(symbol, start=start_date, end=end_date)
            # 매크로는 GLOBAL 폴더에 원자적 저장
            self._save_atomic("GLOBAL", name, df[['Close']])

if __name__ == "__main__":
    collector = IntegratedAtomicCollector()
    today = datetime.now().strftime("%Y%m%d")
    
    # 1. 매크로 수집
    collector.collect_global_macro("2024-01-01", datetime.now().strftime("%Y-%m-%d"))
    # 2. 전 종목 기본 데이터 수집
    collector.collect_market_basic(today, today)
    # 3. 스캘핑 종목 집중 수집
    collector.collect_scalping_micros(today)
