# src/config/trading_config.py

class TradingConfig:
    # 자산 구간 기준 (단위: 원)
    SCALPING_THRESHOLD = 1000000  # 100만 원
    STABLE_THRESHOLD = 2000000    # 200만 원

    # 거래 비용
    TAX_RATE = 0.0018             # 0.18%
    COMMISSION_RATE = 0.00015     # 0.015%
    SLIPPAGE = 0.0005             # 0.05%
