import os
import torch
from src.data.chronological_loader import ChronologicalDataLoader
from src.utils.archiver import AutoArchiver
from src.environment.trading_gym_env import TradingGymEnv
from src.models.dqn_trading_model import DQNAgent
from src.trading.trading_strategy_agent import AssetAllocationAgent

def main():
    # 1. 초기화 (데이터 로더, 아카이버, 전략 에이전트)
    loader = ChronologicalDataLoader()
    archiver = AutoArchiver()
    strategy_agent = AssetAllocationAgent(current_seed=50000) # 5만 원 시작
    
    # 2. AI 모델 및 환경 설정
    env = TradingGymEnv() 
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    
    current_year = None
    print("🏁 [정석 학습] 5년치 1분봉 연대기적 학습을 시작합니다.")

    # 3. 타임라인 순차 학습 루프 (2021 -> 2026)
    for date_str in loader.timeline:
        year = date_str[:4]
        
        # 연도가 바뀌면 이전 연도 데이터 압축 (용량 확보)
        if current_year is not None and current_year != year:
            print(f"\n--- 📅 {current_year}년 학습 완료 ---")
            archiver.compress_year_section(current_year)
            # 모델 체크포인트 저장
            agent.save(f"./checkpoints/model_{current_year}.pth")
        
        current_year = year

        # 해당 날짜의 2,700개 종목 1분봉 병렬 로드 (990 Pro 풀가동)
        print(f"📅 데이터 로드 중: {date_str}...", end="\r")
        day_data = loader.get_day_data_parallel(date_str)
        
        if not day_data:
            continue

        # 4. 학습 실행 (강화학습 에피소드)
        # 5만 원 시드머니 제약 내에서 1분봉 단위로 매매 학습
        for ticker, df in day_data.items():
            # 전략 에이전트가 5만 원으로 살 수 있는 종목인지 필터링
            if strategy_agent.can_buy('SCALPING', df['price'].iloc[0]):
                state = env.reset(df)
                done = False
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                
                # 배치 학습 진행
                if len(agent.memory) > 64:
                    agent.replay(32)

    # 5. 최종 종료 및 마지막 연도 압축
    if current_year:
        archiver.compress_year_section(current_year)
    print("\n✅ 모든 학습이 완료되었습니다. 5년의 경험치가 모델에 쌓였습니다!")

if __name__ == "__main__":
    main()
