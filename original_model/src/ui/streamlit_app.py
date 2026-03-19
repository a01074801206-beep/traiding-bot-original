import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from src.trading.backtester import MonsterBacktester

# 페이지 설정
st.set_page_config(page_title="Monster_Quant v1.0", layout="wide")

def load_data():
    """32GB RAM에 적재된 최종 결과물 로드"""
    # 5D 텐서 및 모델 예측 결과 로드 (캐시 활용)
    tensor_df = pd.read_parquet("data_cache/monster_5d_tensor.parquet")
    # 실제 운영 시에는 최신 예측 결과 파일(json/csv)을 로드
    return tensor_df

def main():
    st.title("🚀 Monster_Quant: AI Multi-modal Trading System")
    st.sidebar.header("🛠️ System Control")
    
    # 1. 사이드바 - 파이프라인 수동 가동 버튼
    if st.sidebar.button("Run Full Pipeline"):
        st.sidebar.info("데이터 수집 및 학습 시작 (i5-14400 & RTX 4060 가동...)")
        # mq.execute() 호출 로직
    
    # 2. 메인 화면 - 상단 지표 (오늘의 시장 요약)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("KOSPI Status", "2,750.12", "+1.2%")
    col2.metric("NASDAQ (Sync)", "16,420.50", "-0.5%")
    col3.metric("BDI Proxy (HMM)", "1,240", "+4.5%")
    col4.metric("AI Confidence", "88%", "Strong Buy")

    # 3. 데이터 로드
    df = load_data()
    
    # 4. 오늘의 AI 추천 종목 (Top 5)
    st.subheader("🎯 AI Top Picks for Tomorrow")
    # 예시 데이터 (실제로는 모델 예측값 출력)
    picks = pd.DataFrame({
        'Ticker': ['011200', '005930', '000660', '010140', '003670'],
        'Name': ['HMM', '삼성전자', 'SK하이닉스', '삼성중공업', '포스코홀딩스'],
        'Pred_Return': ['+4.2%', '+1.5%', '+2.1%', '+3.8%', '+1.2%'],
        'Main_Factor': ['BDI Index ↑', 'SOXX Index ↑', 'HBM News +', 'Order Wins +', 'USD/KRW ↓']
    })
    st.table(picks)

    # 5. 수익률 그래프 (백테스팅 결과 시각화)
    st.subheader("📈 Backtesting Performance")
    # Plotly를 이용한 인터랙티브 차트
    fig = go.Figure()
    # 임의의 백테스트 데이터 생성 (실제 결과 대입)
    fig.add_trace(go.Scatter(x=pd.date_range("2025-01-01", periods=100), 
                             y=np.cumsum(np.random.normal(0.002, 0.01, 100)) + 1,
                             mode='lines', name='Monster_Strategy'))
    fig.add_trace(go.Scatter(x=pd.date_range("2025-01-01", periods=100), 
                             y=np.cumsum(np.random.normal(0.001, 0.01, 100)) + 1,
                             mode='lines', name='KOSPI Benchmark'))
    
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 6. 5D 텐서 데이터 뷰어 (사용자님이 보고 싶어 하신 엑셀 구조 확인)
    st.subheader("📋 5D Tensor Raw Explorer")
    selected_ticker = st.selectbox("Select Ticker", df['ticker'].unique())
    st.dataframe(df[df['ticker'] == selected_ticker].tail(20))

if __name__ == "__main__":
    main()