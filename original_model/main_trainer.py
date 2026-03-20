import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.data.chronological_loader import ChronologicalDataLoader
from src.environment.trading_gym_env import CustomTradingEnv
from src.models.dqn_trading_model import DQNetwork
from src.utils.paths import CHECKPOINT_DIR
from src.config.feature_config import FeatureConfig
from src.config.trading_config import TradingConfig
from src.data.data_fuser import DataFuser

def main():
    # --- [설정 영역] 나중에 데이터를 추가하고 싶으면 아래 리스트에 'news' 등을 넣으세요 ---
    ACTIVE_DATA_GROUPS = []  # 예: ['news', 'macro'] 추가 시 자동 확장
    # --------------------------------------------------------------------------

    # 1. 초기화 및 데이터 도킹 준비
    loader = ChronologicalDataLoader()
    fuser = DataFuser(data_lake_path=os.path.dirname(loader.timeline[0])) # 990 Pro 경로 자동 인식
    
    # 2. 모델 입구(Input Dim) 자동 결정
    input_size = FeatureConfig.get_input_dim(active_groups=ACTIVE_DATA_GROUPS)
    model = DQNetwork(input_dim=input_size, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 체크포인트 로드 (이어하기)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
    if os.path.exists(latest_ckpt):
        model.load_model(latest_ckpt)
        print("🔄 기존 체크포인트에서 학습을 재개합니다.")

    print(f"📊 시스템 가동 (입력 차원: {input_size}) | 활성 데이터: {ACTIVE_DATA_GROUPS}")

    # 3. 연대기별 학습 루프
    for date_str in loader.timeline:
        # 해당 날짜의 원본 주가 데이터 로드
        day_data_dict = loader.get_day_data(date_str)
        
        for ticker, raw_df in day_data_dict.items():
            # [핵심] 데이터 도킹 스테이션 가동 (추가 데이터 병합)
            fused_df = fuser.fuse_all(raw_df, active_groups=ACTIVE_DATA_GROUPS)
            
            # 환경 설정 (각 종목별 하루치 학습)
            env = CustomTradingEnv(fused_df)
            state = env.reset()
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
                
                # 환경 실행 (자산 100만/200만 돌파 시 보상은 Env 내부에서 TradingConfig 참조)
                next_state, reward, done, _ = env.step(action)
                
                # DQN 학습 로직 (생략된 세부 구현은 이전과 동일)
                # ... 학습 코드 ...
                
                state = next_state
            
            print(f"🚀 [{date_str}] {ticker} 완료 | 잔고: {env.balance:,.0f}원", end="\r")

    # 4. 최종 저장
    model.save_model(latest_ckpt)
    print("\n✅ 모든 학습 및 데이터 도킹 프로세스가 완료되었습니다.")

if __name__ == "__main__":
    main()
