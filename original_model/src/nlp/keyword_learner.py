import pandas as pd
import numpy as np
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class MonsterKeywordLearner:
    def __init__(self, vol_threshold=0.04):
        # 주가가 4% 이상 변동한 날을 '사건 발생일'로 간주
        self.vol_threshold = vol_threshold
        self.stop_words = ['오늘', '내일', '뉴스', '진행', '단독', '특징주', '상승', '하락']

    def extract_nouns(self, text):
        """뉴스 제목에서 유의미한 키워드 추출 (간이 토크나이저)"""
        # 실제 운영 시 KoNLPy(Okt, Mecab)를 사용하여 명사만 추출하는 것이 더 정확합니다.
        words = re.findall(r'[가-힣]{2,}', text) 
        return [w for w in words if w not in self.stop_words]

    def analyze_sector_impact(self, sector_name, market_sub_df, news_sub_df):
        """[섹터별] 주가 변동과 뉴스 단어의 상관관계 분석"""
        impact_counter = Counter()
        
        # 1. 급등락일 필터링 (i5-14400 멀티코어 연산 구간)
        volatile_events = market_sub_df[market_sub_df['change'].abs() >= self.vol_threshold]
        
        for _, event in volatile_events.iterrows():
            # 해당 날짜, 해당 종목의 뉴스 찾기
            target_news = news_sub_df[(news_df['date'] == event['date']) & 
                                      (news_df['ticker'] == event['ticker'])]
            
            # 주가가 올랐으면 단어에 +1, 내렸으면 -1 (가중치 부여)
            weight = 1 if event['change'] > 0 else -1
            
            for title in target_news['title']:
                words = self.extract_nouns(title)
                for word in words:
                    impact_counter[word] += weight

        # 2. 통계적으로 유의미한 상위 20개 단어 추출
        # 결과 예시: [('운임', 45), ('공급과잉', -32), ...]
        return impact_counter.most_common(20)

    def update_processor_rules(self, processor, market_df, news_df, sector_map):
        """추출된 키워드를 NewsProcessor의 실시간 규칙에 주입"""
        print(f"🧬 Monster 자가 학습 엔진 가동: 32GB RAM 내 전수 조사 중...")
        
        sectors = set(sector_map.values())
        for sector in sectors:
            # 해당 섹터에 속한 종목들만 데이터 슬라이싱
            tickers_in_sector = [t for t, s in sector_map.items() if s == sector]
            m_sub = market_df[market_df['ticker'].isin(tickers_in_sector)]
            n_sub = news_df[news_df['ticker'].isin(tickers_in_sector)]
            
            top_keywords = self.analyze_sector_impact(sector, m_sub, n_sub)
            
            # 규칙 업데이트 로직
            if sector not in processor.sector_rules:
                processor.sector_rules[sector] = {}
                
            for word, score in top_keywords:
                # 점수 기반 가중치 변환 (예: 긍정 빈도가 높으면 1.2~1.5배)
                if score > 5: # 최소 빈도 조건
                    processor.sector_rules[sector][word] = min(1.0 + (score/100), 1.8)
                elif score < -5:
                    processor.sector_rules[sector][word] = max(1.0 + (score/100), 0.2)
            
            print(f"✅ {sector} 섹터 학습 완료: {len(processor.sector_rules[sector])}개 규칙 적용")