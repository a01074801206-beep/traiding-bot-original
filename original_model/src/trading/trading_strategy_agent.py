import yaml
from src.utils.paths import CONFIG_PATH

class AssetAllocationAgent:
    def __init__(self, current_seed=50000):
        self.current_seed = current_seed
        self.load_config()

    def load_config(self):
        # 'config.yaml' 직접 입력 대신 CONFIG_PATH 사용
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
