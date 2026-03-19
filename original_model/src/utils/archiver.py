import os
import tarfile
from src.utils.paths import DATA_LAKE_DIR, ARCHIVE_DIR

class AutoArchiver:
    def __init__(self):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)

    def compress_year_section(self, year):
        """해당 연도의 모든 1분봉 데이터를 압축하고 원본 삭제"""
        target_file = os.path.join(ARCHIVE_DIR, f"trained_{year}.tar.gz")
        
        print(f"🗜️ {year}년 데이터 압축 중... (990 Pro 용량 확보)")
        with tarfile.open(target_file, "w:gz") as tar:
            for ticker in os.listdir(DATA_LAKE_DIR):
                ticker_path = os.path.join(DATA_LAKE_DIR, ticker)
                for f in os.listdir(ticker_path):
                    if f.startswith(str(year)):
                        full_path = os.path.join(ticker_path, f)
                        tar.add(full_path, arcname=os.path.join(ticker, f))
                        os.remove(full_path) # 압축 후 원본 삭제
        print(f"✅ {year}년 아카이브 완료: {target_file}")
