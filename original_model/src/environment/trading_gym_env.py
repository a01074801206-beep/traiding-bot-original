# 기존 하드코딩: pd.read_parquet(f'./data_lake/{ticker}/...') -> 삭제
# 교정: loader가 준 df를 그대로 받음
class TradingGymEnv:
    def __init__(self):
        self.df = None

    def reset(self, df_from_loader):
        """이제 경로는 로더가 책임지고, 환경은 데이터만 받습니다."""
        self.df = df_from_loader
        self.current_step = 0
        return self._get_observation()
