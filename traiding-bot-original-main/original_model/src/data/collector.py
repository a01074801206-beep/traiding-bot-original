import os
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from src.data.inventory_db import InventoryDB
from src.config.trading_config import TradingConfig

class DataCollector:
    def __init__(self, save_path="D:/trading_data"):
        self.save_path = save_path
        self.db = InventoryDB()
        os.makedirs(self.save_path, exist_ok=True)

    def collect_all_stocks(self, start_date="20210101", end_date=None):
        """전체 종목의 5년치 OHLCV 데이터를 수집"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        # 1. 현재 상장된 모든 종목 리스트 가져오기 (KOSPI + KOSDAQ)
        tickers = stock.get_market_ticker_list(market="ALL")
        
        # 2. DB를 조회해서 이미 완료된 종목 제외 (중복 작업 방지)
        pending_tickers = self.db.get_incomplete_tickers(tickers)
        print(f"🚀 수집 시작: 전체 {len(tickers)}개 중 {len(pending_tickers)}개 진행")

        for ticker in pending_tickers:
            try:
                name = stock.get_market_ticker_name(ticker)
                # 3. 주가 데이터 다운로드 (1분봉은 증권사 API 필요, 여기선 일봉 기준 예시)
                df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
                
                if df.empty:
                    continue

                # 4. 990 Pro에 Parquet 형태로 저장 (압축률 및 속도 최적화)
                ticker_dir = os.path.join(self.save_path, ticker)
                os.makedirs(ticker_dir, exist_ok=True)
                
                # 날짜별로 저장하거나 통째로 저장 (5년치 일봉은 통째가 유리)
                file_path = os.path.join(ticker_dir, f"{ticker}_daily.parquet")
                df.to_parquet(file_path)

                # 5. Inventory DB에 수집 현황 마킹
                self.db.update_stock_status(
                    ticker=ticker,
                    start=start_date,
                    end=end_date,
                    rows=len(df)
                )
                print(f"✅ {name}({ticker}) 수집 완료: {len(df)}행 저장")

            except Exception as e:
                print(f"❌ {ticker} 수집 중 에러 발생: {e}")

    def collect_macro_data(self):
        """환율, 나스닥 등 거시경제 지표 수집 슬롯 (도킹용)"""
        # FinanceDataReader 등을 활용해 환율, 지수 수집 로직 추가 가능
        print("🌐 거시경제 지표 수집을 준비 중입니다...")
        # 수집 후 self.db.mark_date_data(date, 'macro') 호출
