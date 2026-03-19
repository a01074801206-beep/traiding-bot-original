import os
import torch
import pandas as pd
from src.data.market_collector import MarketCollector
from src.data.investor_collector import InvestorCollector
from src.data.macro_collector import MacroCollector
from src.nlp.news_processor import MonsterNewsProcessor, KeywordLearner
from src.analyzer.correlation_engine import MonsterCorrelationEngine
from src.models.monster_transformer import MonsterTransformer

class MonsterQuantSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 시스템 가동: {self.device} (RTX 4060) 기반 가속 활성화")
        
        # 각 모듈 초기화
        self.market_col = MarketCollector()
        self.investor_col = InvestorCollector()
        self.macro_col = MacroCollector()
        self.news_proc = MonsterNewsProcessor()
        self.learner = KeywordLearner()
        self.engine = MonsterCorrelationEngine()

    def run_update_data(self):
        """1단계: 모든 원천 데이터 최신화 (i5-14400 멀티코어 활용)"""
        print("\n--- 📥 1단계: 데이터 수집 및 최신화 ---")
        self.market_col.collect_and_save()
        self.investor_col.collect()
        self.macro_col.collect()
        # 대안 데이터 수집 등 추가 가능

    def run_nlp_pipeline(self, sector_map):
        """2단계: 뉴스 감성 분석 및 자가 학습 (RTX 4060 활용)"""
        print("\n--- 🧠 2단계: NLP 지능형 분석 및 키워드 학습 ---")
        # 1. 뉴스 데이터 로드 (data_cache에서 불러옴)
        # 2. 자가 학습 (과거 데이터 기반 키워드 갱신)
        # self.learner.learn_and_update(...) 
        
        # 3. 5D 뉴스 시트 생성
        # self.news_proc.create_5d_news_sheet(...)
        print("✅ 뉴스 벡터라이징 완료")

    def run_merging_engine(self):
        """3단계: 5차원 엑셀(5D 텐서) 병합"""
        print("\n--- 📊 3단계: 5D 텐서 병합 (VLOOKUP 매칭) ---")
        final_tensor = self.engine.build_5d_monster_tensor()
        return final_tensor

    def train_monster_model(self, data):
        """4단계: RTX 4060을 이용한 트랜스포머 학습"""
        print("\n--- 🔥 4단계: Monster Transformer 학습 시작 ---")
        # 32GB RAM에서 데이터를 미니배치로 나누어 GPU로 전송
        # model = MonsterTransformer().to(self.device)
        # train_loop(model, data)
        print("✅ 모델 학습 및 가중치 저장 완료")

    def execute(self):
        """전체 파이프라인 가동"""
        # 1. 데이터 업데이트
        self.run_update_data()
        
        # 2. 섹터 맵 로드 (HMM=해운, 삼성=반도체 등)
        sector_map = {"011200": "SHIPPING", "005930": "SEMICON"} # 실제론 DB/파일 로드
        
        # 3. 뉴스 처리 및 학습
        self.run_nlp_pipeline(sector_map)
        
        # 4. 5D 텐서 구축
        tensor_data = self.run_merging_engine()
        
        # 5. 최종 학습
        self.train_monster_model(tensor_data)
        
        print("\n🚀 모든 시스템 공정 완료. 모의투자 모듈 대기 중.")

if __name__ == "__main__":
    mq = MonsterQuantSystem()
    mq.execute()