# src/config/feature_config.py

class FeatureConfig:
    # 1. 기본 주가 데이터 (현재 사용 중)
    BASE_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'vwap']
    
    # 2. 계좌 상태 데이터 (현재 사용 중)
    STATE_FEATURES = ['balance', 'holdings_ratio', 'pnl', 'time_left']
    
    # 3. 미래 확장 데이터 슬롯 (추가 대기 중)
    EXTENDED_FEATURES = {
        "news": ["sentiment_score", "news_count"],      # 뉴스 감성 점수 등
        "macro": ["usd_krw", "kospi_index", "interest"], # 환율, 지수, 금리
        "supply": ["foreign_buy", "inst_buy"]           # 수급 데이터
    }

    @classmethod
    def get_input_dim(cls, active_groups=[]):
        """현재 활성화된 데이터 그룹에 따라 모델 입구(Input Dim) 크기 계산"""
        total_dim = len(cls.BASE_FEATURES) + len(cls.STATE_FEATURES)
        for group in active_groups:
            if group in cls.EXTENDED_FEATURES:
                total_dim += len(cls.EXTENDED_FEATURES[group])
        return total_dim
