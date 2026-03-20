# src/data/data_fuser.py
import pandas as pd
import os

class DataFuser:
    def __init__(self, data_lake_path):
        self.path = data_lake_path

    def fuse_all(self, price_df, active_groups=[]):
        """주가 데이터프레임에 추가 데이터를 옆으로 이어붙임 (Merge)"""
        fused_df = price_df.copy()
        
        # 1. 뉴스 데이터 병합 (파일이 있을 때만)
        if "news" in active_groups:
            news_path = os.path.join(self.path, "news_processed.parquet")
            if os.path.exists(news_path):
                news_df = pd.read_parquet(news_path)
                fused_df = pd.merge(fused_df, news_df, on='date', how='left').fillna(0)

        # 2. 거시 지표 병합
        if "macro" in active_groups:
            macro_path = os.path.join(self.path, "macro_indicators.parquet")
            if os.path.exists(macro_path):
                macro_df = pd.read_parquet(macro_path)
                fused_df = pd.merge(fused_df, macro_df, on='date', how='left').ffill()
        
        return fused_df
