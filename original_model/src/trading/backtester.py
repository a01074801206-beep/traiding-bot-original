import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MonsterBacktester:
    def __init__(self, initial_capital=10_000_000): # 기본 자금 1,000만원
        self.capital = initial_capital
        self.fee = 0.00015 # 수수료 약 0.015%
        self.tax = 0.0020  # 매도 세금 약 0.2%
        self.portfolio = {} # 현재 보유 종목 {ticker: quantity}
        
    def run_simulation(self, test_df, model_predictions, top_k=5):
        """
        [백테스팅 실행]
        test_df: 날짜별 실제 주가 데이터
        model_predictions: 모델이 예측한 종목별 수익률 {date: {ticker: pred_return}}
        top_k: 매일 매수할 상위 종목 수
        """
        print("📈 백테스팅 시뮬레이션 시작...")
        history = []
        dates = sorted(test_df['date'].unique())
        
        current_cash = self.capital

        for i in range(len(dates) - 1):
            today = dates[i]
            tomorrow = dates[i+1]
            
            # 1. 모델의 오늘 예측값 가져오기
            preds = model_predictions.get(today, {})
            if not preds: continue
            
            # 2. 수익률 상위 K개 종목 선정 (롱 전략)
            # 예측 수익률이 0보다 큰 종목들만 필터링
            sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
            target_tickers = [t for t, p in sorted_preds[:top_k] if p > 0]
            
            # 3. 리밸런싱 (전량 매도 후 재매수 가정 - 단순화 버전)
            # 실제 구현 시 보유 종목 유지 로직을 넣으면 수수료가 절감됨
            if target_tickers:
                each_invest = current_cash / len(target_tickers)
                daily_return = 0
                
                for ticker in target_tickers:
                    # 실제 내일의 수익률 계산 (5D 텐서에서 가져옴)
                    actual_ret = test_df[(test_df['date'] == tomorrow) & 
                                         (test_df['ticker'] == ticker)]['change'].values
                    
                    if len(actual_ret) > 0:
                        # 수익률 합산 (수수료/세금 차감)
                        net_ret = actual_ret[0] - (self.fee * 2 + self.tax)
                        daily_return += net_ret / len(target_tickers)
                
                current_cash *= (1 + daily_return)
            
            history.append({'date': tomorrow, 'capital': current_cash})

        return pd.DataFrame(history)

    def analyze_performance(self, history_df):
        """성과 지표 계산 (MDD, Sharpe Ratio 등)"""
        df = history_df.copy()
        df['return'] = df['capital'].pct_change()
        
        cumulative_return = (df['capital'].iloc[-1] / self.capital - 1) * 100
        daily_std = df['return'].std()
        sharpe_ratio = (df['return'].mean() / daily_std) * np.sqrt(252) if daily_std != 0 else 0
        
        # MDD 계산
        df['peak'] = df['capital'].cummax()
        df['drawdown'] = (df['capital'] - df['peak']) / self.peak
        mdd = df['drawdown'].min() * 100
        
        print(f"\n=== 🏆 전략 성과 분석 결과 ===")
        print(f"누적 수익률: {cumulative_return:.2f}%")
        print(f"최대 낙폭(MDD): {mdd:.2f}%")
        print(f"샤프 지수: {sharpe_ratio:.2f}")
        
        return df