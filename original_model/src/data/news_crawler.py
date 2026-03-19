import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

class PastNewsCrawler:
    def __init__(self):
        self.base_url = "https://search.naver.com/search.naver?where=news&query={query}&ds={start}&de={end}"

    def fetch_past_news(self, ticker_name, target_date):
        """특정 종목의 특정 날짜 뉴스를 크롤링 (API 없이 수집)"""
        date_str = target_date.strftime("%Y.%m.%d")
        url = self.base_url.format(query=ticker_name, start=date_str, end=date_str)
        
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        titles = [title.text for title in soup.select("a.news_tit")]
        return titles

    def collect_history(self, ticker_list, days=365):
        """최근 1년치 뉴스를 종목별로 전수 조사 (노동 집약적 작업)"""
        # i5-14400의 멀티프로세싱을 사용하여 날짜별로 수집하면 빠릅니다.
        pass