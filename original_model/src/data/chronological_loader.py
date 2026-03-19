import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.utils.paths import DATA_LAKE_DIR # 경로 관리 파일 참조

class ChronologicalDataLoader:
    def __init__(self):
        self.base_path = DATA_LAKE_DIR
        self.tickers = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        self.timeline = self._build_timeline()

    def _build_timeline(self):
        """데이터 레이크를 훑어 전체 학습 날짜 리스트 생성 (2021 -> 2026)"""
        all_dates = set()
        # 모든 종목을 다 뒤지면 느리니 상위 20개 종목에서 날짜 추출
        for ticker in self.tickers[:20]:
            path = os.path.join(self.base_path, ticker)
            files = [f.split('_')[0] for f in os.listdir(path) if f.endswith('.parquet')]
            all_dates.update(files)
        return sorted(list(all_dates))

    def _load_single_parquet(self, ticker, date_str):
        """개별 종목의 1분봉 파일을 읽어오는 내부 함수"""
        file_path = os.path.join(self.base_path, ticker, f"{date_str}_1min.parquet")
        if os.path.exists(file_path):
            # 메모리 절약을 위해 float32 사용
            return ticker, pd.read_parquet(file_path).astype({'price':'float32', 'volume':'int32'})
        return ticker, None

    def get_day_data_parallel(self, date_str):
        """990 Pro의 I/O를 활용해 2,700개 종목 데이터를 병렬로 로드"""
        day_data = {}
        with ThreadPoolExecutor(max_workers=16) as executor: # CPU 코어 활용
            results = executor.map(lambda t: self._load_single_parquet(t, date_str), self.tickers)
            for ticker, df in results:
                if df is not None:
                    day_data[ticker] = df
        return day_data
