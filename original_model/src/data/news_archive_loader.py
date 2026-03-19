import pandas as pd
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class NewsArchiveLoader:
    def __init__(self, raw_dir="data_raw/bigkinds"):
        self.raw_dir = raw_dir
        self.output_path = "data_cache/news_final_sheet.parquet"
        self.cpu_cores = 10

    def load_and_clean(self):
        """빅카인즈에서 받은 여러 CSV 파일 통합"""
        print("📂 빅카인즈 과거 뉴스 데이터 통합 중...")
        all_files = glob.glob(os.path.join(self.raw_dir, "*.xlsx")) # 빅카인즈는 보통 xlsx 제공
        
        df_list = []
        for file in all_files:
            # 필요한 컬럼만 추출 (일자, 제목, 키워드, 특성추출(가중치순))
            tmp = pd.read_excel(file, usecols=['일자', '제목', '특성추출(가중치순)'])
            df_list.append(tmp)
            
        full_df = pd.concat(df_list).drop_duplicates()
        full_df['date'] = pd.to_datetime(full_df['일자'], format='%Y%m%d')
        return full_df

    def _tag_ticker_to_news(self, news_chunk, ticker_dict):
        """뉴스와 종목을 매칭 (뉴스 제목에 종목명이나 관련 키워드가 있는지 검사)"""
        tagged_results = []
        for _, row in news_chunk.iterrows():
            content = str(row['제목']) + " " + str(row['특성추출(가중치순)'])
            
            for ticker, name in ticker_dict.items():
                if name in content:
                    tagged_results.append({
                        'date': row['date'],
                        'ticker': ticker,
                        'title': row['제목']
                    })
        return pd.DataFrame(tagged_results)

    def build_history_sheet(self, ticker_dict):
        """전체 뉴스를 종목별로 매핑하여 5D 텐서용 뉴스 시트 생성"""
        full_df = self.load_and_clean()
        
        # 데이터를 10개 뭉치로 나눠서 i5-14400의 10개 코어에 할당
        chunks = np.array_split(full_df, self.cpu_cores)
        
        print(f"🔥 i5-14400 가동: 과거 뉴스 {len(full_df)}건 종목 매칭 시작...")
        with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
            results = list(tqdm(executor.map(self._tag_ticker_to_news, chunks, 
                                            [ticker_dict]*len(chunks)), 
                                total=len(chunks)))
            
        final_news_sheet = pd.concat(results)
        final_news_sheet.to_parquet(self.output_path, compression='snappy')
        print(f"✅ 과거 뉴스 매칭 완료: {self.output_path}")
        return final_news_sheet