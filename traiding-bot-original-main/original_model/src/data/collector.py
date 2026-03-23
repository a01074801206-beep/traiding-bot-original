import os
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime
from tqdm import tqdm
from src.data.inventory_db import InventoryDB

class DataCollector:
    def __init__(self, base_path="D:/trading_data"):
        """
        base_path: 990 Pro SSD 경로 (성능 극대화를 위해 SSD 권장)
        """
        self.base_path = base_path
        self.macro_path = os.path.join(base_path, "macro")
        self.db = InventoryDB()
        
        # 폴더 구조 초기화
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.macro_path, exist_ok=True)

    def collect_all(self, start_date="20210101"):
        """
        통합 수집 실행: 거시지표 -> 섹터지도 -> 전종목 주가/수급
        """
        print("\n" + "="*60)
        print(f"🚀 [시스템 가동] 데이터 수집 및 990 Pro 적재 시작")
        print("="*60)
        
        # 1. 거시 경제 지표 (나스닥, VIX, 환율 등)
        self._collect_macro_indices(start_date)
        
        # 2. 전 종목 섹터 매핑 (테마 분류용)
        self._collect_sector_map()
        
        # 3. 개별 종목 주가 및 수급 (핵심 데이터)
        self._collect_stock_ohlcv(start_date)
        
        print("\n" + "="*60)
        print(f"✨ [수집 완료] 모든 데이터가 {self.base_path}에 저장되었습니다.")
        print("="*60)

    def _collect_macro_indices(self, start_date):
        print(f"\n📡 [Step 1/3] 거시 지표 수집 및 도킹 준비...")
        indices = {
            'KS11': 'KOSPI',
            'KQ11': 'KOSDAQ',
            'IXIC': 'NASDAQ',
            'VIX': 'VIX',
            'USD/KRW': 'FX_USD',
            'US10YT=X': 'US_10Y'
        }
        
        for symbol, name in indices.items():
            try:
                df = fdr.DataReader(symbol, start_date)
                if not df.empty:
                    save_file = os.path.join(self.macro_path, f"{name}.parquet")
                    df.to_parquet(save_file, engine='pyarrow')
                    self.db.mark_date_data(df.index[-1].strftime("%Y%m%d"), f"macro_{name}")
                    print(f"  ✅ {name} 수집 완료 ({len(df)}행)")
            except Exception as e:
                print(f"  ❌ {name} 수집 실패: {e}")

    def _collect_sector_map(self):
        print(f"\n📂 [Step 2/3] 전 종목 섹터 및 업종 지도 생성...")
        try:
            df_krx = fdr.StockListing('KRX')
            df_sectors = df_krx[['Symbol', 'Name', 'Sector', 'Industry']]
            save_file = os.path.join(self.base_path, "sector_map.parquet")
            df_sectors.to_parquet(save_file, engine='pyarrow')
            print(f"  ✅ {len(df_sectors)}개 종목 매핑 데이터 저장 완료")
        except Exception as e:
            print(f"  ❌ 섹터 수집 실패: {e}")

    def _collect_stock_ohlcv(self, start_date):
        """
        tqdm 진행률 바를 포함한 주가/수급 통합 수집
        """
        end_date = datetime.now().strftime("%Y%m%d")
        all_tickers = stock.get_market_ticker_list(market="ALL")
        
        # DB 확인 후 미수집 종목만 추출
        pending_tickers = self.db.get_incomplete_tickers(all_tickers)
        total_count = len(pending_tickers)
        
        print(f"\n📈 [Step 3/3] 주가 및 수급 데이터 수집 (대상: {total_count}개 종목)")
        
        # 진행바 설정
        pbar = tqdm(pending_tickers, total=total_count, desc="전체 진행률", unit="종목")
        
        for ticker in pbar:
            try:
                # 1. 주가 (OHLCV)
                df_price = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
                if df_price.empty:
                    continue
                
                # 2. 수급 (외인, 기관, 개인 순매수)
                df_investor = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, ticker)
                
                # 3. 데이터 결합 (Column 방향으로 합치기)
                df_combined = pd.concat([df_price, df_investor], axis=1)
                
                # 4. 990 Pro 전용 폴더 저장
                ticker_dir = os.path.join(self.base_path, ticker)
                os.makedirs(ticker_dir, exist_ok=True)
                
                save_file = os.path.join(ticker_dir, f"{ticker}_daily.parquet")
                df_combined.to_parquet(save_file, engine='pyarrow', compression='snappy')
                
                # 5. Inventory DB 상태 업데이트
                self.db.update_stock_status(
                    ticker=ticker,
                    start=start_date,
                    end=end_date,
                    rows=len(df_combined)
                )
                
                # 진행바 하단 상태 메시지 업데이트
                pbar.set_postfix(ticker=ticker, speed="High-I/O")
                
            except Exception as e:
                # 에러 발생 시 로그만 남기고 다음 종목으로 진행
                continue

if __name__ == "__main__":
    # 필수 패키지 설치 확인용: pip install pykrx finance-datareader pandas pyarrow tqdm
    collector = DataCollector()
    collector.collect_all()
