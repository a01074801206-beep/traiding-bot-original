import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class FeatureProcessor:
    def __init__(self, base_path="D:/trading_data"):
        self.base_path = base_path

    def add_technical_indicators(self, df):
        """데이터프레임에 보조 지표 주입"""
        # 1. 이동평균선 (Moving Average)
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        df['MA60'] = df['종가'].rolling(window=60).mean()

        # 2. RSI (Relative Strength Index) - 과매수/과매도 지표
        delta = df['종가'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 3. 볼린저 밴드 (Bollinger Bands) - 변동성 지표
        df['std'] = df['종가'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['std'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['std'] * 2)

        # 4. 거래량 이동평균
        df['VMA5'] = df['거래량'].rolling(window=5).mean()
        
        # 5. 수익률 (로그 수익률) - AI 학습에 더 적합함
        df['Return'] = np.log(df['종가'] / df['종가'].shift(1))
        
        # NaN 값 제거 (이동평균 계산 시 발생하는 초기 빈칸 삭제)
        return df.dropna()

    def process_all_stocks(self):
        # D:/trading_data 내의 모든 종목 폴더 리스트업
        tickers = [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f)) and f != 'macro']
        
        print(f"\n🚀 [Feature Engineering] 2,879개 종목 보조지표 도킹 시작")
        pbar = tqdm(tickers, desc="990 Pro 연산 중")

        for ticker in pbar:
            try:
                file_path = os.path.join(self.base_path, ticker, f"{ticker}_daily.parquet")
                if not os.path.exists(file_path):
                    continue

                # 데이터 로드
                df = pd.read_parquet(file_path)
                
                # 지표 추가
                df_featured = self.add_technical_indicators(df)
                
                # 990 Pro에 덮어쓰기 (또는 _featured.parquet으로 저장 가능)
                df_featured.to_parquet(file_path, engine='pyarrow')
                
                pbar.set_postfix(ticker=ticker, final_cols=len(df_featured.columns))
            except Exception as e:
                continue

        print("\n✨ 모든 종목에 보조 지표 주입이 완료되었습니다!")

if __name__ == "__main__":
    processor = FeatureProcessor()
    processor.process_all_stocks()
