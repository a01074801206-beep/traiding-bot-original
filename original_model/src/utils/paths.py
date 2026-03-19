import os

# 현재 파일(src/utils/paths.py) 위치 기준, 최상위 루트 폴더(traiding-bot-original/)를 잡습니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 990 Pro SSD 내 2,700개 종목 폴더가 담긴 곳
DATA_LAKE_DIR = os.path.join(BASE_DIR, "data_lake")

# 학습 완료 후 압축 파일이 저장될 곳
ARCHIVE_DIR = os.path.join(BASE_DIR, "archives")

# AI 모델 가중치 저장소
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# 설정 파일 경로
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# 폴더 자동 생성 (없을 경우 대비)
for p in [ARCHIVE_DIR, CHECKPOINT_DIR]:
    os.makedirs(p, exist_ok=True)
