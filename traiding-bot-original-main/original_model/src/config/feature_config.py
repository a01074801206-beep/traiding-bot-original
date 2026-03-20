# src/config/feature_config.py

class FeatureConfig:
    # 1. 미시 데이터 (Micro / Intra-day)
    MICRO_DATA = ['tick_data', 'order_imbalance', 'cancel_ratio', 'order_strength', 'large_order_count']
    
    # 2. 시장 수급 데이터 (Supply / Demand)
    SUPPLY_DATA = ['investor_trend', 'foreign_ratio_change', 'short_stock_balance', 'loan_balance', 'deposit_amount']
    
    # 3. 글로벌 & 거시 경제 (Global / Macro)
    MACRO_DATA = ['nasdaq100', 'phlx_semiconductor', 'buffett_indicator', 'fear_greed_index', 'usd_krw_vol', 'yield_curve_gap']
    
    # 4. 대체 데이터 (Alternative Data)
    ALT_DATA = ['news_embedding', 'disclosure_analysis', 'patent_status', 'hiring_data', 'factory_utilization', 'esg_score']
    
    # 5. 물류 및 공급망 (Supply Chain)
    LOGISTICS_DATA = ['export_import_stats', 'bdi_scfi_index', 'port_congestion']

    @classmethod
    def get_input_dim(cls, active_groups=[]):
        # 기본 OHLCV + 기술적 지표 + 상태값 (약 20개 가정)
        base_dim = 20 
        # 선택된 그룹의 데이터 개수만큼 입구(Input) 확장
        # (각 리스트의 길이를 합산하는 로직)
        return base_dim + 추가된_데이터_개수
