import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re

class UniversalNewsProcessor:
    """
    2,700개 전 종목의 뉴스를 긁어 감성 점수(4D 레이어)를 계산하고
    원자적 방식(1지표 1파일)으로 종목 폴더에 배포함.
    """
    def __init__(self, base_path="./data_lake"):
        self.base_path = base_path
        self.master_path = os.path.join(base_path, "MASTER", "ticker_master.parquet")
        
        if not os.path.exists(self.master_path):
            raise Exception("❗ [오류] 마스터 맵이 없습니다. master_mapper.py를 먼저 실행하세요.")
        
        self.master_df = pd.read_parquet(self.master_path)
        
        # 1. 간단한 주식 전문 감성 사전 (lexicon) 정의 (고도화 가능)
        self.lexicon = {
            # 긍정 단어 (+1)
            '상승': 1.0, '호재': 1.0, '급등': 1.2, '최고': 1.0, '수주': 1.2, '흑자': 1.1, '성장': 1.0, '목표가 상향': 1.3,
            # 부정 단어 (-1)
            '하락': -1.0, '악재': -1.0, '급락': -1.2, '최저': -1.0, '손실': -1.1, '적자': -1.1, '우려': -1.0, '목표가 하향': -1.3
        }

    def _crawl_news_titles(self, ticker, target_date):
        """네이버 금융에서 특정 종목의 당일 뉴스 제목 크롤링"""
        # (실제 구현 시 Fake User-Agent 및 프록시 권장)
        url = f"https://finance.naver.com/item/news_news.naver?code={ticker}"
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            titles = []
            for title_html in soup.select('.title'):
                titles.append(title_html.get_text(strip=True))
            return titles
        except Exception:
            return []

    def _calculate_sentiment(self, titles):
        """뉴스 제목 리스트의 평균 감성 점수 계산 (-2.0 ~ +2.0)"""
        if not titles: return 0.0
        
        total_score = 0
        count = 0
        for title in titles:
            title_score = 0
            # 사전 기반 매칭
            for word, score in self.lexicon.items():
                if word in title:
                    title_score += score
            
            if title_score != 0:
                total_score += title_score
                count += 1
        
        return total_score / count if count > 0 else 0.0

    def _save_atomic(self, ticker, feature_name, data):
        """감성 점수를 종목 폴더에 원자적으로 저장 (Append 로직 포함)"""
        if data is None or data.empty: return
        target_dir = os.path.join(self.base_path, ticker)
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, f"{feature_name}.parquet")
        
        if os.path.exists(file_path):
            old_df = pd.read_parquet(file_path)
            # 중복 제거 (Timestamp 기준)
            combined_df = pd.concat([old_df, data]).drop_duplicates(subset=['timestamp'])
            combined_df.to_parquet(file_path)
        else:
            data.to_parquet(file_path)

    def process_daily_news(self, target_date=None):
        """전 종목 뉴스를 가공하여 4D 레이어 배포"""
        if not target_date:
            target_date = datetime.now().strftime("%Y%m%d")
        
        print(f">>> [{target_date}] 전 종목 뉴스 감성 분석 시작...")
        
        # 2. 마스터 맵에서 테마 정보 추출 (스마트 배포용 사전 구축)
        # 예: {'반도체': ['005930', '000660'], '2차전지': ['005490']}
        theme_map = {}
        # (실제 구현 시 master_df에 테마 컬럼이 있다고 가정)
        # if 'theme' in self.master_df.columns:
        #     for _, row in self.master_df.iterrows():
        #         for theme in row['theme']:
        #             theme_map.setdefault(theme, []).append(row['ticker'])

        for ticker in self.master_df['ticker']:
            # 3. 개별 종목 뉴스 크롤링 및 점수 계산
            titles = self._crawl_news_titles(ticker, target_date)
            score = self._calculate_sentiment(titles)
            
            # 4. 개별 종목 폴더에 저장 (1지표 1파일: news_sentiment.parquet)
            news_df = pd.DataFrame({
                'timestamp': [datetime.now()], # 수집 시점 저장
                'sentiment': [score]
            })
            self._save_atomic(ticker, "news_sentiment", news_df)
            
            # 5. [4D 확장] 테마 뉴스 스마트 배포 (옵션)
            # if 'theme' in self.master_df.columns:
            #     # 해당 종목의 테마 정보를 읽어, 동일 테마 종목들에게도 점수 배포
            #     themes = self.master_df[self.master_df['ticker'] == ticker]['theme'].iloc[0]
            #     for theme in themes:
            #         for peer_ticker in theme_map.get(theme, []):
            #             if peer_ticker != ticker:
            #                 self._save_atomic(peer_ticker, f"theme_news_{theme}", news_df)
            
            if self.master_df['ticker'].tolist().index(ticker) % 100 == 0:
                print(f"   [News OK] {self.master_df['ticker'].tolist().index(ticker)} / {len(self.master_df)} 완료")
            time.sleep(0.1) # 서버 부하 방지

        print(">>> 모든 종목의 뉴스 4D 레이어 가공 및 원자적 저장 완료.")

if __name__ == "__main__":
    processor = UniversalNewsProcessor(base_path="./data_lake")
    processor.process_daily_news()
