import os

# Root: traiding-bot-original/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_LAKE_DIR = os.path.join(BASE_DIR, "data_lake")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archives")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# 폴더 자동 생성
for p in [ARCHIVE_DIR, CHECKPOINT_DIR]:
    os.makedirs(p, exist_ok=True)
