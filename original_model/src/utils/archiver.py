import os, tarfile
from src.utils.paths import DATA_LAKE_DIR, ARCHIVE_DIR

class AutoArchiver:
    def compress_year_section(self, year):
        target = os.path.join(ARCHIVE_DIR, f"trained_{year}.tar.gz")
        print(f"📦 {year}년 데이터 압축 중...")
        with tarfile.open(target, "w:gz") as tar:
            for ticker in os.listdir(DATA_LAKE_DIR):
                t_path = os.path.join(DATA_LAKE_DIR, ticker)
                for f in os.listdir(t_path):
                    if f.startswith(str(year)):
                        tar.add(os.path.join(t_path, f), arcname=os.path.join(ticker, f))
                        os.remove(os.path.join(t_path, f))
