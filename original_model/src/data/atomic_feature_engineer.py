import os
import pandas as pd
# 만약 utils에 경로 설정 파일을 만드셨다면:
# from src.utils.paths import DATA_LAKE_DIR

class AtomicFeatureEngineer:
    """
    수집된 원자적 데이터를 읽어 기술적 지표(Feature)를 생성함.
    결과물 역시 1지표 1파일 원칙에 따라 종목 폴더에 저장.
    """
    def __init__(self, base_path="./data_lake"):
        self.base_path = base_path
        self.master_path = os.path.join(base_path, "MASTER", "ticker_master.parquet")
        
        if not os.path.exists(self.master_path):
            raise Exception("❗ [오류] 마스터 맵이 없습니다. master_mapper.py를 먼저 실행하세요.")
        
        self.master_df = pd.read_parquet(self.master_path)
        self.tickers = self.master_df['ticker'].tolist()

    def _load_atomic(self, ticker, feature_name):
        """종목 폴더에서 특정 지표 파일 로드"""
        file_path = os.path.join(self.base_path, ticker, f"{feature_name}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return None

    def _save_atomic(self, ticker, feature_name, data):
        """계산된 지표를 종목 폴더에 저장"""
        if data is None or data.empty: return
        target_path = os.path.join(self.base_path, ticker, f"{feature_name}.parquet")
        data.to_parquet(target_path)

    def process_technical_indicators(self):
        """전 종목을 순회하며 낙주/스캘핑용 지표 계산"""
        print(f">>> 지표 가공 시작 (대상: {len(self.tickers)}개 종목)...")
        
        for ticker in self.tickers:
            # 1. 기초 데이터 로드 (종가 기준)
            df_close = self._load_atomic(ticker, "close")
            if df_close is None or len(df_close) < 30: continue
            
            # 종가 컬럼명 통일
            df_close.columns = ['close']
            
            # --- [낙주 매매 15% 슬롯용 지표] ---
            # RSI (과매도 포착용)
            delta = df_close['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            self._save_atomic(ticker, "rsi_14", rsi.to_frame(name="rsi"))

            # 볼린저 밴드 (하단 이탈 포착용)
            ma20 = df_close['close'].rolling(window=20).mean()
            std20 = df_close['close'].rolling(window=20).std()
            bb_low = ma20 - (std20 * 2)
            self._save_atomic(ticker, "bb_low", bb_low.to_frame(name="bb_low"))

            # --- [스캘핑 20% 슬롯용 지표] ---
            # 변동성 (ATR 대용 간단 지표)
            volatility = df_close['close'].pct_change().rolling(window=10).std()
            self._save_atomic(ticker, "price_volatility", volatility.to_frame(name="std_10"))

            # 이동평균선 괴리율 (단기 슈팅/과락 확인)
            disparity = (df_close['close'] / ma20) * 100
            self._save_atomic(ticker, "ma20_disparity", disparity.to_frame(name="disparity"))

            if self.tickers.index(ticker) % 100 == 0:
                print(f"   [Processing] {self.tickers.index(ticker)} / {len(self.tickers)} 완료")

        print(">>> 모든 종목의 기술적 지표 가공 및 원자적 저장 완료.")

if __name__ == "__main__":
    engineer = AtomicFeatureEngineer(base_path="./data_lake")
    engineer.process_technical_indicators()
