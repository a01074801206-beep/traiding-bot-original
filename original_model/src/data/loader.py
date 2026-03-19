import pandas as pd
import gc

class DataLoader:
    def __init__(self):
        self.paths = {
            "market": "data_cache/market_raw.parquet",
            "investor": "data_cache/investor_raw.parquet",
            "macro": "data_cache/macro_raw.parquet",
            "alt": "data_cache/alt_raw.parquet"
        }

    def load_merged_data(self, ticker):
        """특정 종목에 대해 모든 데이터(시장+수급+매크로+대안)를 결합하여 리턴"""
        # 32GB RAM을 아끼기 위해 필요한 종목만 필터링해서 Join
        market = pd.read_parquet(self.paths["market"], filters=[('ticker', '==', ticker)])
        investor = pd.read_parquet(self.paths["investor"], filters=[('ticker', '==', ticker)])
        macro = pd.read_parquet(self.paths["macro"])
        alt = pd.read_parquet(self.paths["alt"])
        
        # 날짜 기준으로 결합
        df = market.join(investor.drop(columns='ticker'), how='left')
        df = df.join(macro, how='left').join(alt, how='left')
        
        return df.ffill()

if __name__ == "__main__":
    # 사용 예시
    loader = DataLoader()
    hmm_data = loader.load_merged_data("011200") # HMM 종목코드
    print(hmm_data.tail())