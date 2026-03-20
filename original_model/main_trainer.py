import os
import torch
import torch.optim as optim
from src.data.chronological_loader import ChronologicalDataLoader
from src.environment.trading_gym_env import CustomTradingEnv
from src.models.dqn_trading_model import DQNetwork
from src.utils.paths import CHECKPOINT_DIR, ARCHIVE_DIR
from src.utils.archiver import AutoArchiver

def main():
    # 1. 초기화 (데이터 로더, 모델, 최적화 도구)
    loader = ChronologicalDataLoader()
    archiver = AutoArchiver()
    
    # 모델 생성 (입력 15, 출력 3)
    model = DQNetwork(input_dim=15, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 기존 학습 데이터가 있다면 로드 (이어하기)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
    if os.path.exists(latest_ckpt):
        model.load_model(latest_ckpt)
        print("🔄 기존 체크포인트에서 학습을 재개합니다.")

    current_year = None
    env = None

    print("📊 5년치 연대기 학습 시스템 가동...")

    # 2. 날짜별 루프 (990 Pro 고속 로딩 활용)
    for date_str in loader.timeline:
        year = date_str[:4]
        
        # 연도가 바뀌면 자동 압축 (용량 확보)
        if current_year and current_year != year:
            archiver.compress_year_section(current_year)
            model.save_model(os.path.join(CHECKPOINT_DIR, f"model_{current_year}.pth"))
        
        current_year = year
        
        # 해당 날짜의 2,700개 종목 데이터 로드
        day_data_dict = loader.get_day_data(date_str)
        
        for ticker, df in day_data_dict.items():
            # 환경 설정 (각 종목별 하루치 학습)
            if env is None:
                env = CustomTradingEnv(df)
            state = env.reset(df)
            done = False
            
            while not done:
                # AI의 결정 (Epsilon-greedy 생략, 핵심 로직 위주)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
                
                # 환경 실행 (자산 100만/200만 돌파 시 보상 체계 자동 변경됨)
                next_state, reward, done, _ = env.step(action)
                
                # 학습 업데이트 (간략화된 DQN 로직)
                # 실제 구현 시에는 Replay Buffer와 Target Network가 추가됩니다.
                target = reward + (0.99 * torch.max(model(torch.FloatTensor(next_state).unsqueeze(0))))
                loss = F.mse_loss(model(state_tensor), target.unsqueeze(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                state = next_state
            
            # 실시간 로그 출력
            print(f"🚀 [{date_str}] {ticker} 학습 완료 | 잔고: {env.balance:,.0f}원", end="\r")

    # 3. 최종 저장
    model.save_model(latest_ckpt)
    archiver.compress_year_section(current_year)
    print("\n✅ 모든 학습이 완료되었습니다. 990 Pro 시스템을 종료합니다.")

if __name__ == "__main__":
    main()
