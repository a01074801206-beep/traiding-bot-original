from src.data.chronological_loader import ChronologicalDataLoader
from src.utils.archiver import AutoArchiver

def main():
    loader = ChronologicalDataLoader()
    archiver = AutoArchiver()
    current_year = None

    for date_str in loader.timeline:
        year = date_str[:4]
        if current_year and current_year != year:
            archiver.compress_year_section(current_year)
        
        current_year = year
        day_data = loader.get_day_data(date_str)
        
        # 여기서 AI 모델 학습(train) 코드 실행
        print(f"🚀 학습 중: {date_str} ({len(day_data)} 종목)", end="\r")

    archiver.compress_year_section(current_year)

if __name__ == "__main__":
    main()
