import os
from src.data.chronological_loader import ChronologicalDataLoader
from src.utils.archiver import AutoArchiver # 아카이버 모듈
from src.utils.paths import CHECKPOINT_DIR

def main():
    loader = ChronologicalDataLoader()
    archiver = AutoArchiver() # 학습 완료 데이터 압축 도구
    
    current_year = None
    print(f"🚀 5년치 1분봉 정석 학습 시작 (총 {len(loader.timeline)}일 분량)")

    for date_str in loader.timeline:
        year = date_str[:4]
        
        # 연도가 바뀌는 시점에 이전 연도 데이터 압축/삭제
        if current_year and current_year != year:
            print(f"\n📦 {current_year}년 학습 완료! 990 Pro 용량 확보를 위해 압축을 시작합니다.")
            archiver.compress_year_section(current_year)
        
        current_year = year
        
        # 1분봉 데이터 로드 및 모델 학습 (생략된 학습 로직 실행)
        day_data = loader.get_day_data(date_str)
        # TODO: model.train(day_data) 
        
        print(f"📅 학습 중: {date_str} ({len(day_data)} 종목 완료)", end="\r")

    # 마지막 연도 처리
    archiver.compress_year_section(current_year)
    print("\n✅ 모든 연대기적 학습이 완료되었습니다.")

if __name__ == "__main__":
    main()
