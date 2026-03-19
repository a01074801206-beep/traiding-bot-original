import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class MacroCollector:
    def __init__(self, days=1825):
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        self.save_path = "data_cache/macro_raw.parquet"
        
        # 5D 텐서 매칭을 위한 핵심 심볼 정의
        self.symbols = {
            "^IXIC": "nasdaq",          # 나스닥 지수 (기술주 동조화)
            "SOXX": "phil_semicon",     # 필라델피아 반도체 (삼성/SK용)
            "KRW=X": "usd_krw",         # 원/달러 환율 (수출주용)
            "BDRY": "bdi_proxy",        # BDI 운임지수 프록시 (HMM용)
            "^TNX": "us_10y_yield",     # 미국 10년물 금리 (성장주 밸류용)
            "CL=F": "wti_oil",          # WTI 원유 선물 (에너지/정유용)
            "^VIX": "vix"               # 공포 지수 (시장 변동성)
        }

    def collect(self):
        print(f"🌐 글로벌 매크로 및 섹터 지표 수집 시작 (Ticker: {len(self.symbols)}개)...")
        
        try:
            # 1. yfinance를 이용한 일괄 다운로드 (멀티스레딩 기본 지원)
            data = yf.download(
                list(self.symbols.keys()), 
                start=self.start_date, 
                end=self.end_date,
                threads=True
            )['Close']
            
            # 2. 컬럼명 가독성 있게 변경
            data = data.rename(columns=self.symbols)
            
            # 3. 시계열 데이터 정제 (주말/휴장일 결측치 처리)
            # 32GB RAM 연산을 위해 전방/후방 채우기(ffill/bfill) 수행
            data = data.ffill().bfill()
            
            # 4. 데이터 타입 최적화 (float32) 및 인덱스 정리
            data = data.astype('float32').reset_index()
            data.columns = [c.lower() for c in data.columns] # 컬럼명 소문자 통일
            
            # 5. 저장
            data.to_parquet(self.save_path, compression='snappy')
            print(f"✅ 매크로 데이터 저장 완료: {self.save_path} ({len(data)}일치)")
            
            return data
            
        except Exception as e:
            print(f"❌ 매크로 수집 중 오류 발생: {e}")
            return None

if __name__ == "__main__":
    collector = MacroCollector()
    collector.collect()