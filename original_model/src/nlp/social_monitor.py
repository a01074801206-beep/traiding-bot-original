import pandas as pd

class SocialMonitor:
    def __init__(self):
        self.keywords = ["급등", "상따", "풀매수", "탈출", "구조대", "한강"]

    def analyze_hype(self, text_list):
        """커뮤니티 광기 지수(Hype Index) 계산"""
        hype_count = 0
        for text in text_list:
            for word in self.keywords:
                if word in text:
                    hype_count += 1
        return hype_count / len(text_list) if text_list else 0

    # 실제 구현 시 텔레그램 채널이나 특정 커뮤니티 크롤링 로직 추가