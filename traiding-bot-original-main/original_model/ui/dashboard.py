import streamlit as st
import pandas as pd
import time
import os
import plotly.express as px
from src.config.trading_config import TradingConfig

# 페이지 설정
st.set_page_config(page_title="AI Trading Bot Monitor", layout="wide")

st.title("🚀 실시간 AI 트레이딩 학습 관제탑")

# 좌측 사이드바: 시스템 상태
st.sidebar.header("🖥️ 시스템 상태")
st.sidebar.info(f"대상 종목 수: 2,700개\n데이터 경로: D:/trading_data (990 Pro)")
st.sidebar.divider()
st.sidebar.write(f"**수익 구간 설정**")
st.sidebar.write(f"- 스캘핑: ~{TradingConfig.SCALPING_THRESHOLD:,}원")
st.sidebar.write(f"- 안정화: {TradingConfig.STABLE_THRESHOLD:,}원~")

# 메인 화면 레이아웃
col1, col2 = st.columns(2)

def load_data():
    # 학습 로그 파일(CSV/Parquet)을 읽어오는 로직
    # main_trainer.py에서 학습 결과가 저장되는 경로를 지정하세요.
    log_path = "logs/training_log.csv"
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame()

# 리프레시 루프
placeholder = st.empty()

while True:
    df = load_data()
    
    with placeholder.container():
        if not df.empty:
            # 1. 상단 주요 지표 (Key Metrics)
            m1, m2, m3, m4 = st.columns(4)
            latest = df.iloc[-1]
            m1.metric("현재 잔고", f"{latest['balance']:,}원", f"{latest['return']:.2%}")
            m2.metric("학습 진행도", f"{latest['progress']:.1%}")
            m3.metric("현재 학습 종목", latest['ticker'])
            m4.metric("평균 Loss", f"{latest['loss']:.4f}")

            # 2. 수익률 곡선 그래프
            with col1:
                st.subheader("📈 누적 수익률 추이")
                fig_rev = px.line(df, x=df.index, y="balance", template="plotly_dark")
                st.plotly_chart(fig_rev, use_container_width=True)

            # 3. 모델 손실(Loss) 및 과적합 감시
            with col2:
                st.subheader("📉 학습 손실 (Loss Curve)")
                fig_loss = px.line(df, x=df.index, y="loss", color_discrete_sequence=['red'])
                st.plotly_chart(fig_loss, use_container_width=True)

            # 4. 최근 거래 내역 테이블
            st.subheader("📝 최근 학습 거래 로그")
            st.dataframe(df.tail(10), use_container_width=True)
            
        else:
            st.warning("⚠️ 아직 학습 로그 데이터가 없습니다. main_trainer.py를 먼저 실행하세요.")
    
    time.sleep(2) # 2초마다 갱신
