import yfinance as yf
import pandas as pd

class AlternativeCollector:
    def __init__(self):
        self.cache_path = "data_cache/alt_raw.parquet"

    def collect(self):
        print("🚢 대안 데이터(BDI 프록시, 섹터 지표) 수집 중...")
        # BDI 운임지수 대용인 BDRY ETF 활용
        bdry = yf.download("BDRY", start="2020-01-01")['Close']
        bdry.name = "bdi_proxy"
        
        # 추가적인 섹터 데이터(예: 반도체 ETF, 에너지 ETF 등)를 여기에 확장
        alt_df = pd.DataFrame(bdry).ffill()
        alt_df.to_parquet(self.cache_path)
        print(f"✅ 대안 데이터 저장 완료: {self.cache_path}")

if __name__ == "__main__":
    AlternativeCollector().collect()