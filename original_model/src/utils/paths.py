import os

# 프로젝트 루트 (traiding-bot-original/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 데이터 저장소 (2,700개 종목 폴더가 있는 곳)
DATA_LAKE_DIR = os.path.join(BASE_DIR, "data_lake")

# 아카이브 저장소 (학습 완료된 압축 파일)
ARCHIVE_DIR = os.path.join(BASE_DIR, "archives")

# 모델 저장소
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# 필요한 폴더 자동 생성
for path in [ARCHIVE_DIR, CHECKPOINT_DIR]:
    os.makedirs(path, exist_ok=True)
