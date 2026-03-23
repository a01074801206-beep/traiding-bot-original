import os
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime
from src.data.inventory_db import InventoryDB

class DataCollector:
    def __init__(self, base_path="D:/trading_data"):
        self.base_path = base_path
        self.macro_path = os.path.join(base_path, "macro")
        self.db = InventoryDB()
        
        # 폴더 생성
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.macro_path, exist_ok=True)

    def collect_all(self, start_date="20210101"):
        """
        [통합 실행] 1.시장지표 -> 2.섹터지도 -> 3.전종목 주가 수집
        """
        print("🏁 통합 수집 프로세스 시작...")
        
        # 1. 시장 지표 및 공포지수 수집
        self._collect_macro_indices(start_date)
        
        # 2. 섹터(테마) 지도 생성
        self._collect_sector_map()
        
        # 3. 개별 종목 주가(OHLCV) 수집
        self._collect_stock_ohlcv(start_date)
        
        print("\n🚀 모든 데이터 수집 및 990 Pro 저장 완료!")

    def _collect_macro_indices(self, start_date):
        """시장 지표 (나스닥, VIX, 환율, 국채금리 등)"""
        print("🌐 시장 지표(Macro) 수집 중...")
        indices = {
            'KS11': 'KOSPI',
            'KQ11': 'KOSDAQ',
            'IXIC': 'NASDAQ',
            'VIX': 'VIX',         # 공포지수
            'USD/KRW': 'FX_USD',   # 환율
            'US10YT=X': 'US_10Y'   # 미국 10년물 국채금리
        }
        
        for symbol, name in indices.items():
            try:
                df = fdr.DataReader(symbol, start_date)
                if not df.empty:
                    file_path = os.path.join(self.macro_path, f"{name}.parquet")
                    df.to_parquet(file_path)
                    # DB에 마크 (최신 날짜 기준)
                    last_date = df.index[-1].strftime("%Y%m%d")
                    self.db.mark_date_data(last_date, f"macro_{name}")
                    print(f"  - {name} 수집 완료")
            except Exception as e:
                print(f"  - {name} 수집 실패: {e}")

    def _collect_sector_map(self):
        """2,700개 종목의 섹터/업종 매핑 데이터"""
        print("📁 섹터 및 테마 정보 수집 중...")
        try:
            df_krx = fdr.StockListing('KRX')
            # 필요한 컬럼만 추출 (종목코드, 이름, 업종, 주요제품)
            df_sectors = df_krx[['Symbol', 'Name', 'Sector', 'Industry']]
            file_path = os.path.join(self.base_path, "sector_map.parquet")
            df_sectors.to_parquet(file_path)
            print(f"  - {len(df_sectors)}개 종목 섹터 지도 저장 완료")
        except Exception as e:
            print(f"  - 섹터 수집 실패: {e}")

    def _collect_stock_ohlcv(self, start_date):
        """전 종목 주가 및 수급 데이터"""
        end_date = datetime.now().strftime("%Y%m%d")
        
        # 전 종목 리스트 가져오기
        tickers = stock.get_market_ticker_list(market="ALL")
        
        # DB를 조회해서 이미 완료된 종목 제외 (인벤토리 체크)
        pending_tickers = self.db.get_incomplete_tickers(tickers)
        print(f"📈 주가 데이터 수집 시작 (남은 대상: {len(pending_tickers)}개)")

        for ticker in pending_tickers:
            try:
                # 1. 주가 데이터 (일봉)
                df_ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
                if df_ohlcv.empty: continue
                
                # 2. 투자자별 순매수량 (수급 데이터 도킹용)
                df_investor = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, ticker)
                
                # 주가와 수급 합치기
                df_combined = pd.concat([df_ohlcv, df_investor], axis=1)
                
                # 990 Pro 전용 경로 저장
                ticker_dir = os.path.join(self.base_path, ticker)
                os.makedirs(ticker_dir, exist_ok=True)
                
                file_path = os.path.join(ticker_dir, f"{ticker}_daily.parquet")
                df_combined.to_parquet(file_path)
                
                # Inventory DB 등록
                self.db.update_stock_status(
                    ticker=ticker,
                    start=start_date,
                    end=end_date,
                    rows=len(df_combined)
                )
                print(f"  ✅ {ticker} 완료 ({len(df_combined)}일치)", end="\r")
                
            except Exception as e:
                print(f"\n❌ {ticker} 처리 중 에러: {e}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all()
