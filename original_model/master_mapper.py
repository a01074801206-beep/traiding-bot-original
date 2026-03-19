import os
import pandas as pd
from pykrx import stock
from datetime import datetime

class UniverseMasterMapper:
    """
    2,700개 전 종목을 사용자 전략(스캘핑, 낙주, 안정)에 따라 분류하고
    데이터 레이크 폴더 구조를 생성하는 마스터 클래스
    """
    def __init__(self, base_path="./data_lake"):
        self.base_path = base_path
        # 마스터 데이터 저장 폴더 생성
        os.makedirs(os.path.join(self.base_path, "MASTER"), exist_ok=True)
        # 글로벌/매크로 데이터 폴더 생성
        os.makedirs(os.path.join(self.base_path, "GLOBAL"), exist_ok=True)

    def generate_master_map(self):
        print(">>> [Step 1] 전 종목 시장 데이터 로드 중...")
        # 최근 영업일 기준 데이터 가져오기
        target_date = datetime.now().strftime("%Y%m%d")
        
        # 1. 시세 및 거래대금 정보
        try:
            df_price = stock.get_market_ohlcv_by_ticker(target_date, market="ALL")
            df_cap = stock.get_market_cap_by_ticker(target_date, market="ALL")
        except Exception as e:
            print(f"데이터 로드 실패 (영업일 확인 필요): {e}")
            return None

        # 데이터 병합 (티커 기준)
        master = pd.concat([df_price, df_cap[['시가총액']]], axis=1).reset_index()
        master.rename(columns={'티커': 'ticker'}, inplace=True)
        
        print(">>> [Step 2] 전략 슬롯별 종목 분류 시작...")
        
        # 2. 전략 슬롯 분류 로직 (사용자 자금 운용 전략 기반)
        def classify_strategy(row):
            # A. 스캘핑 (20% 비중): 거래대금 상위 10% 종목 (유동성 확보)
            if row['거래대금'] >= master['거래대금'].quantile(0.9):
                return 'SCALPING_20'
            
            # B. 안정 수익 (65% 비중): 시가총액 상위 20% 우량주 (코스피200급)
            elif row['시가총액'] >= master['시가총액'].quantile(0.8):
                return 'STABLE_65'
            
            # C. 낙주 매매 (15% 비중): 그 외 변동성 매매 가능 종목
            else:
                return 'CONTRARIAN_15'

        master['strategy_slot'] = master.apply(classify_strategy, axis=1)

        # 3. 마스터 파일 저장 (Parquet 형식)
        master_path = os.path.join(self.base_path, "MASTER", "ticker_master.parquet")
        master.to_parquet(master_path)
        print(f">>> [Step 3] 마스터 맵 저장 완료: {master_path}")
        
        # 4. 종목별 독립 폴더 생성 (원자적 구조의 핵심)
        print(f">>> [Step 4] {len(master)}개 종목별 폴더 생성 중...")
        for ticker in master['ticker']:
            ticker_dir = os.path.join(self.base_path, ticker)
            if not os.path.exists(ticker_dir):
                os.makedirs(ticker_dir)
        
        print("="*50)
        print(f"분류 결과 요약:")
        print(master['strategy_slot'].value_counts())
        print("="*50)
        
        return master

if __name__ == "__main__":
    # 데이터 레이크 경로 설정 (집 PC 환경에 맞춰 수정 가능)
    mapper = UniverseMasterMapper(base_path="./data_lake")
    mapper.generate_master_map()
