import streamlit as st
import pandas as pd
import time

def run_dashboard():
    st.set_page_config(page_title="AI Trading Bot Monitor", layout="wide")
    st.title("🚀 Real-time Training Dashboard")

    # 상단 요약 지표
    col1, col2, col3, col4 = st.columns(4)
    balance_metric = col1.empty()
    asset_metric = col2.empty()
    strategy_metric = col3.empty()
    loss_metric = col4.empty()

    # 실시간 그래프 영역
    chart_col1, chart_col2 = st.columns(2)
    equity_chart = chart_col1.empty()
    loss_chart = chart_col2.empty()

    # 데이터 누적용 리스트 (실제로는 DB나 파일을 공유해서 읽음)
    history = {"step": [], "balance": [], "loss": []}

    while True:
        # 990 Pro에 저장된 실시간 로그 파일을 읽어옴
        try:
            log_data = pd.read_csv("logs/live_train_log.csv").tail(100)
            
            # 지표 업데이트
            current_balance = log_data['balance'].iloc[-1]
            balance_metric.metric("Current Balance", f"{current_balance:,.0f}원")
            
            # 전략 단계 판별 (사용자님의 설계 반영)
            strategy = "공격 모드"
            if current_balance >= 2000000: strategy = "안정/복합 모드"
            elif current_balance >= 1000000: strategy = "스캘핑/낙수 모드"
            strategy_metric.metric("Active Strategy", strategy)

            # 그래프 업데이트
            equity_chart.line_chart(log_data.set_index('step')['balance'])
            loss_chart.line_chart(log_data.set_index('step')['loss'])
            
        except:
            st.warning("학습 데이터를 기다리는 중...")
            
        time.sleep(1) # 1초마다 갱신
