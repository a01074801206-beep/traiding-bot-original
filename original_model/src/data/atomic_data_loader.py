import os
import pandas as pd
import numpy as np
from datetime import datetime

class AtomicDataLoader:
    """
    종목 폴더별로 흩어진 원자적 파일들을 행 단위로 정합하여 
    3D(수치), 4D/5D(이벤트) 텐서를 생성하는 로더.
    """
    def __init__(self, base_path="./data_lake"):
        self.base_path = base_path
        self.master_path = os.path.join(base_path, "MASTER", "ticker_master.parquet")
        
        if not os.path.exists(self.master_path):
            raise Exception("❗ 마스터 맵이 없습니다. 이전 단계들을 먼저 완료하세요.")
        
        self.master_df = pd.read_parquet(self.master_path)

    def get_tickers_by_strategy(self, strategy_type):
        """특정 전략 슬롯(SCALPING_20, CONTRARIAN_15, STABLE_65) 종목만 추출"""
        return self.master_df[self.master_df['strategy_slot'] == strategy_type]['ticker'].tolist()

    def assemble_row(self, ticker, target_features):
        """
        특정 종목의 여러 지표 파일들을 '시간' 기준으로 Join하여 하나의 DF로 병합.
        target_features: ['close', 'rsi_14', 'buy_intensity', 'nasdaq'] 등
        """
        combined_df = pd.DataFrame()

        for feature in target_features:
            # 1. 글로벌 지표인지 종목별 지표인지 판단 후 로드
            global_path = os.path.join(self.base_path, "GLOBAL", f"{feature}.parquet")
            local_path = os.path.join(self.base_path, ticker, f"{feature}.parquet")
            
            path = global_path if os.path.exists(global_path) else local_path
            
            if os.path.exists(path):
                feature_df = pd.read_parquet(path)
                # 컬럼명을 지표명으로 통일
                feature_df.columns = [feature]
                
                if combined_df.empty:
                    combined_df = feature_df
                else:
                    # '시간(index)'을 기준으로 병합 (Outer Join으로 누락 방지 후 Fillna)
                    combined_df = combined_df.join(feature_df, how='outer')
        
        # 2. 전처리: 스캘핑/낙주 매매는 시계열 연속성이 중요하므로 선형 보간 또는 직전값 채우기
        combined_df = combined_df.ffill().dropna()
        return combined_df

    def create_batch_tensor(self, strategy_slot, feature_list, window_size=30):
        """
        특정 전략에 맞는 종목들의 데이터를 묶어 학습용 3D 텐서 생성
        (Samples, Time_Steps, Features)
        """
        tickers = self.get_tickers_by_strategy(strategy_slot)
        all_data = []
        
        print(f">>> [{strategy_slot}] 텐서 조립 중... (대상 종목: {len(tickers)}개)")
        
        for ticker in tickers[:50]: # 테스트를 위해 우선 50개 종목만 샘플링
            df = self.assemble_row(ticker, feature_list)
            if len(df) < window_size: continue
            
            # 윈도우 슬라이싱 (강화학습/LSTM 입력용)
            values = df.values
            for i in range(len(values) - window_size):
                all_data.append(values[i : i + window_size])
        
        return np.array(all_data)

if __name__ == "__main__":
    loader = AtomicDataLoader()
    
    # 예시: 스캘핑(20%) 전략을 위한 데이터 조립 (Z축 지표 선택)
    scalping_features = ['close', 'volume', 'price_volatility', 'ma20_disparity', 'nasdaq']
    tensor = loader.create_batch_tensor('SCALPING_20', scalping_features)
    
    print(f"조립된 텐서 형태 (Samples, Time, Features): {tensor.shape}")
