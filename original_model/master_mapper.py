import os
from src.utils.paths import DATA_LAKE_DIR # 하드코딩 제거

def create_ticker_folders(tickers):
    for ticker in tickers:
        # 경로를 직접 타이핑하지 않고 변수를 사용합니다.
        target_path = os.path.join(DATA_LAKE_DIR, ticker)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            print(f"✅ 폴더 생성 완료: {target_path}")
