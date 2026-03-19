import os
import requests
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class MonsterNewsProcessor:
    def __init__(self):
# config.yaml 로드
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.client_id = config['API_KEYS']['NAVER_CLIENT_ID']
        self.client_secret = config['API_KEYS']['NAVER_CLIENT_SECRET']
        # 1. RTX 4060 가속 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "kriss-p/ko-finbert" # 금융 특화 BERT
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval() # 추론 모드

        # 2. 섹터별 인과관계 사전 (사용자 정의 '반대 해석' 로직)
        self.sector_rules = {
            "SHIPPING": {"운임 상승": 1.5, "유가 상승": 0.3, "홍해": 1.2},
            "ENERGY": {"원전": 1.8, "SMR": 1.6, "천연가스 상승": 1.2},
            "SEMICON": {"HBM": 2.0, "수율": 1.5, "나스닥 상승": 1.1}
        }
        
        # API 설정 (환경변수나 config에서 관리 권장)
        self.client_id = "YOUR_NAVER_CLIENT_ID"
        self.client_secret = "YOUR_NAVER_CLIENT_SECRET"

    def fetch_news_api(self, query):
        """네이버 뉴스 검색 API 호출 (I/O 작업)"""
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {"X-Naver-Client-Id": self.client_id, "X-Naver-Client-Secret": self.client_secret}
        params = {"query": query, "display": 20, "sort": "sim"}
        
        try:
            res = requests.get(url, headers=headers, params=params)
            return res.json().get('items', [])
        except:
            return []

    def get_sentiment_score(self, text):
        """BERT 모델을 이용한 딥러닝 감성 추출 (RTX 4060 활용)"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Softmax를 통해 긍정 확률(index 1) 추출
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()

    def process_ticker(self, ticker, ticker_name, sector):
        """[개별 종목] 뉴스 수집 및 5D 벡터 생성"""
        items = self.fetch_news_api(ticker_name)
        if not items:
            return [0.5, 0, 0, 0.5, 0.5] # 뉴스 없을 시 중립 데이터

        scores = []
        for item in items:
            title = item['title'].replace("<b>", "").replace("</b>", "")
            base_score = self.get_sentiment_score(title)
            
            # 섹터별 가중치 보정 (HMM에게 유가 상승은 점수 삭감 등)
            multiplier = 1.0
            if sector in self.sector_rules:
                for kw, weight in self.sector_rules[sector].items():
                    if kw in title:
                        multiplier *= weight
            
            final_score = np.clip(base_score * multiplier, 0, 1)
            scores.append(final_score)

        # 5차원 뉴스 피처 (이 숫자들이 엑셀 열이 됨)
        return [
            np.mean(scores),          # 평균 감성
            np.std(scores),           # 감성 변동성
            min(len(items)/20, 1),    # 화제성 (Hype)
            np.max(scores),           # 최대 호재
            np.min(scores)            # 최대 악재
        ]

    def run_daily_analysis(self, ticker_list, sector_map):
        """전 종목 대상 뉴스 분석 실행"""
        print(f"🧠 RTX 4060 가동: {len(ticker_list)}개 종목 뉴스 벡터화 시작...")
        
        results = []
        # API 호출은 I/O 병목이므로 ThreadPool 사용
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 실무에서는 API 속도 제한(Rate Limit)을 고려하여 딜레이를 주거나 쪼개서 호출
            futures = [executor.submit(self.process_ticker, t, n, sector_map.get(t, 'DEFAULT')) 
                       for t, n in ticker_list.items()]
            
            for i, f in enumerate(tqdm(futures)):
                results.append(f.result())
        
        return results

if __name__ == "__main__":
    # 테스트 코드
    proc = MonsterNewsProcessor()
    test_tickers = {"011200": "HMM", "005930": "삼성전자"}
    test_sectors = {"011200": "SHIPPING", "005930": "SEMICON"}
    
    output = proc.run_daily_analysis(test_tickers, test_sectors)
    print(output)