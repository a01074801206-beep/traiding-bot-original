import os
import pandas as pd

class DataInventoryManager:
    def __init__(self, data_lake_path):
        self.base_path = data_lake_path # D:/trading_data/ (990 Pro 경로)
        
    def get_pending_tickers(self, total_ticker_list):
        """
        전체 종목 리스트 중에서 아직 데이터(Parquet)가 생성되지 않은 종목만 반환
        """
        # 1. 이미 데이터가 존재하는 폴더/파일 리스트 추출
        existing_tickers = [
            f for f in os.listdir(self.base_path) 
            if os.path.isdir(os.path.join(self.base_path, f))
        ]
        
        # 2. 전체 리스트에서 기존 리스트 차집합 구하기
        pending_tickers = list(set(total_ticker_list) - set(existing_tickers))
        
        print(f"✅ 총 종목: {len(total_ticker_list)} | 이미 완료: {len(existing_tickers)} | 남은 작업: {len(pending_tickers)}")
        return pending_tickers

    def check_data_completeness(self, ticker, target_days=1250):
        """
        특정 종목의 데이터가 5년치(약 1250일)가 다 찼는지 정밀 검사
        """
        ticker_path = os.path.join(self.base_path, ticker)
        if not os.path.exists(ticker_path):
            return False
            
        files = [f for f in os.listdir(ticker_path) if f.endswith('.parquet')]
        return len(files) >= target_days
