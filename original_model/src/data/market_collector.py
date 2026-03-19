import os
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class MarketCollector:
    def __init__(self, days=1825): # 5년치 기본 설정
        self.end_date = datetime.now().strftime("%Y%m%d")
        self.start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        self.save_path = "data_cache/market_raw.parquet"
        self.cpu_cores = 10  # i5-14400의 코어 수 활용
        
        if not os.path.exists("data_cache"):
            os.makedirs("data_cache")

    def get_tickers(self):
        """코스피, 코스닥 종목 리스트 통합"""
        kospi = stock.get_market_ticker_list(self.end_date, market="KOSPI")
        kosdaq = stock.get_market_ticker_list(self.end_date, market="KOSDAQ")
        return kospi + kosdaq

    def fetch_ohlcv(self, ticker):
        """[개별 종목] 가격 및 거래량 수집 (병렬 처리용)"""
        try:
            df = stock.get_market_ohlcv_by_date(self.start_date, self.end_date, ticker)
            if df.empty: return None
            
            # 종목 코드 및 추가 정보 삽입
            df['ticker'] = ticker
            df = df.reset_index()
            
            # 32GB RAM 효율을 위한 메모리 최적화 (float64 -> float32)
            cols = ['시가', '고가', '저가', '종가', '거래량']
            df[cols] = df[cols].astype('float32')
            return df
        except Exception:
            return None

    def collect(self):
        tickers = self.get_tickers()
        print(f"🚀 i5-14400 엔진 가동: {len(tickers)}개 종목 OHLCV 수집 시작...")

        # ProcessPoolExecutor를 사용하여 CPU 코어 10개 풀가동
        with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
            results = list(tqdm(executor.map(self.fetch_ohlcv, tickers), total=len(tickers)))

        # 데이터 통합 및 저장
        final_df = pd.concat([r for r in results if r is not None])
        
        # 엑셀처럼 정렬하기 위해 컬럼명 영문 변환 (모델 입력용)
        final_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'change', 'ticker']
        
        final_df.to_parquet(self.save_path, compression='snappy')
        print(f"✅ 저장 완료: {self.save_path} (총 {len(final_df)}행)")

if __name__ == "__main__":
    collector = MarketCollector()
    collector.collect()