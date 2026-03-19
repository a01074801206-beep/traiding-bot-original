import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.utils.paths import DATA_LAKE_DIR

class ChronologicalDataLoader:
    def __init__(self):
        self.base_path = DATA_LAKE_DIR
        self.tickers = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        self.timeline = self._build_timeline()

    def _build_timeline(self):
        all_dates = set()
        for ticker in self.tickers[:10]: # 샘플링
            path = os.path.join(self.base_path, ticker)
            files = [f.split('_')[0] for f in os.listdir(path) if f.endswith('.parquet')]
            all_dates.update(files)
        return sorted(list(all_dates))

    def _load_unit(self, ticker, date_str):
        file_path = os.path.join(self.base_path, ticker, f"{date_str}_1min.parquet")
        if os.path.exists(file_path):
            return ticker, pd.read_parquet(file_path).astype({'price':'float32', 'volume':'int32'})
        return ticker, None

    def get_day_data(self, date_str):
        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(lambda t: self._load_unit(t, date_str), self.tickers))
        return {t: df for t, df in results if df is not None}
