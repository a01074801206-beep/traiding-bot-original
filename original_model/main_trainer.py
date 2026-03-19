from src.environment.trading_gym_env import TradingGymEnv
from src.models.dqn_trading_model import DQNAgent
import torch

def start_training(strategy='SCALPING_20', episodes=100):
    env = TradingGymEnv(strategy_slot=strategy)
    # Observation shape: (window_size, features)
    sample_obs = env.reset()
    agent = DQNAgent(state_dim=sample_obs.shape[0], feature_dim=sample_obs.shape[1])

    print(f"🚀 [{strategy}] 전략 학습 시작...")

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, done))
            agent.train_step(batch_size=64)
            
            state = next_state
            total_reward += reward

        # 타겟 모델 업데이트
        if ep % 5 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            print(f"Episode: {ep}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Balance: {info['balance']:.0f}")

    # 학습된 모델 저장
    torch.save(agent.model.state_dict(), f"model_{strategy}.pth")
    print("✅ 학습 완료 및 모델 저장됨.")

if __name__ == "__main__":
    # 5만 원 시드머니의 20%를 담당하는 스캘핑 모델 학습 시작
    start_training(strategy='SCALPING_20', episodes=50)
