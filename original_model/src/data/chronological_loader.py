import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.utils.paths import DATA_LAKE_DIR

class ChronologicalDataLoader:
    def __init__(self):
        self.base_path = DATA_LAKE_DIR
        # 폴더 내 종목 코드 리스트 추출
        self.tickers = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        self.timeline = self._build_timeline()

    def _build_timeline(self):
        """전체 종목 중 대표 종목들의 파일명을 분석해 학습 날짜 순서를 정렬합니다."""
        all_dates = set()
        for ticker in self.tickers[:10]: # 상위 10개 종목 기준
            path = os.path.join(self.base_path, ticker)
            files = [f.split('_')[0] for f in os.listdir(path) if f.endswith('.parquet')]
            all_dates.update(files)
        return sorted(list(all_dates)) # 2021 -> 2026 순서 보장

    def _load_unit(self, ticker, date_str):
        file_path = os.path.join(self.base_path, ticker, f"{date_str}_1min.parquet")
        if os.path.exists(file_path):
            # 메모리 효율을 위해 float32 변환
            df = pd.read_parquet(file_path).astype({'price': 'float32', 'volume': 'int32'})
            return ticker, df
        return ticker, None

    def get_day_data(self, date_str):
        """해당 날짜의 전 종목 1분봉을 990 Pro에서 병렬로 퍼올립니다."""
        day_results = {}
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self._load_unit, t, date_str) for t in self.tickers]
            for f in futures:
                t, df = f.result()
                if df is not None:
                    day_results[t] = df
        return day_results
