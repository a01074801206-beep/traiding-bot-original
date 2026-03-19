import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class MonsterCorrelationEngine:
    def __init__(self):
        self.cache_dir = "data_cache"
        self.output_path = os.path.join(self.cache_dir, "monster_5d_tensor.parquet")
        self.cpu_cores = 10 # i5-14400 10코어 활용
        
        # 32GB RAM 점유율 최적화를 위한 데이터 타입 강제 정의
        self.dtypes = {
            'ticker': 'category', 
            'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
            'volume': 'float32', 'for_net': 'float32', 'inst_net': 'float32',
            'nasdaq': 'float32', 'bdi_proxy': 'float32', 'usd_krw': 'float32',
            'sent_avg': 'float32', 'hype_idx': 'float32'
        }

    def load_all_fragments(self):
        """저장된 모든 데이터 조각 로드"""
        print("📂 32GB RAM에 데이터 조각 적재 시작...")
        market = pd.read_parquet(f"{self.cache_dir}/market_raw.parquet")
        investor = pd.read_parquet(f"{self.cache_dir}/investor_raw.parquet")
        macro = pd.read_parquet(f"{self.cache_dir}/macro_raw.parquet")
        news = pd.read_parquet(f"{self.cache_dir}/news_final_sheet.parquet")
        
        # 날짜 형식 통일 (매칭의 핵심 키)
        for df in [market, investor, macro, news]:
            df['date'] = pd.to_datetime(df['date'])
            
        return market, investor, macro, news

    def _align_single_ticker(self, ticker, market, investor, macro, news):
        """[개별 종목] 모든 차원을 날짜 기준으로 정렬 및 병합 (VLOOKUP 역할)"""
        try:
            # 1. 해당 종목 데이터 슬라이싱
            t_market = market[market['ticker'] == ticker].sort_values('date')
            t_investor = investor[investor['ticker'] == ticker].sort_values('date')
            t_news = news[news['ticker'] == ticker].sort_values('date')
            
            if t_market.empty: return None

            # 2. 고속 시계열 병합 (merge_asof)
            # 가격 + 수급 병합
            merged = pd.merge_asof(t_market, t_investor.drop(columns='ticker'), 
                                    on='date', direction='backward')
            
            # + 매크로(나스닥, BDI 등) 병합
            merged = pd.merge_asof(merged, macro, on='date', direction='backward')
            
            # + 뉴스 감성 수치 병합
            merged = pd.merge_asof(merged, t_news.drop(columns='ticker'), 
                                    on='date', direction='backward')

            # 3. 추가 피처 엔지니어링 (상관관계 지표)
            # 예: 나스닥과의 20일 상관계수 (동조화 여부)
            merged['nasdaq_corr'] = merged['close'].rolling(20).corr(merged['nasdaq'])
            
            # 결측치 처리 (주말 등으로 인한 빈칸 채우기)
            merged = merged.ffill().bfill().astype(self.dtypes, errors='ignore')
            
            return merged
        except:
            return None

    def build_5d_tensor(self):
        """전체 종목 병합 및 최종 5D 텐서 파일 생성"""
        market, investor, macro, news = self.load_all_fragments()
        tickers = market['ticker'].unique()
        
        print(f"🔥 i5-14400 병합 엔진 가동: {len(tickers)}개 종목 시계열 정렬 중...")
        
        # 10개 코어를 활용한 병렬 병합
        with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
            results = list(tqdm(executor.map(self._align_single_ticker, tickers, 
                                            [market]*len(tickers), [investor]*len(tickers), 
                                            [macro]*len(tickers), [news]*len(tickers)), 
                                total=len(tickers)))
            
        # 결과 통합
        final_tensor = pd.concat([r for r in results if r is not None])
        
        # 최종 저장 (Snappy Parquet)
        final_tensor.to_parquet(self.output_path, compression='snappy')
        print(f"✅ 5D 텐서 구축 완료: {self.output_path}")
        return final_tensor

if __name__ == "__main__":
    engine = MonsterCorrelationEngine()
    engine.build_5d_tensor()