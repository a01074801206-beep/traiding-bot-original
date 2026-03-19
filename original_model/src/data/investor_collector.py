import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class InvestorCollector:
    def __init__(self, days=1825):
        self.end_date = datetime.now().strftime("%Y%m%d")
        self.start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        self.save_path = "data_cache/investor_raw.parquet"

    def fetch_investor(self, ticker):
        """[개별 종목] 투자자별 순매수 데이터 수집"""
        try:
            # 외인, 기관, 개인 순매수량
            df = stock.get_market_net_purchases_of_equities_by_ticker(self.start_date, self.end_date, ticker)
            if df.empty: return None
            
            # 공매도 잔고 추가 (필요 시)
            # short = stock.get_exhaustion_rates_of_foreign_investment_by_date(self.start_date, self.end_date, ticker)
            
            df['ticker'] = ticker
            return df.reset_index()
        except:
            return None

    def collect(self):
        tickers = stock.get_market_ticker_list(self.end_date, market="ALL")
        print(f"📊 수급 데이터 수집 시작...")

        with ProcessPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(executor.map(self.fetch_investor, tickers), total=len(tickers)))

        final_df = pd.concat([r for r in results if r is not None])
        final_df.columns = ['date', 'for_net', 'inst_net', 'ind_net', 'etc_net', 'total_net', 'ticker']
        final_df.to_parquet(self.save_path, compression='snappy')
        print(f"✅ 수급 데이터 저장 완료: {self.save_path}")