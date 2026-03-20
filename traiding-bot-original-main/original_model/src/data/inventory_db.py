import sqlite3
import os
from datetime import datetime

class InventoryDB:
    def __init__(self, db_path="inventory.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        # 종목별 기본 주가 데이터 현황
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_master (
                ticker TEXT PRIMARY KEY,
                start_date TEXT,
                end_date TEXT,
                total_rows INTEGER,
                last_updated TIMESTAMP
            )
        ''')
        
        # 추가 데이터(뉴스, 거시 등) 수집 현황
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_availability (
                date TEXT PRIMARY KEY,
                has_news INTEGER DEFAULT 0,
                has_macro INTEGER DEFAULT 0,
                has_supply INTEGER DEFAULT 0,
                has_orderbook INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()

    def update_stock_status(self, ticker, start, end, rows):
        """종목별 주가 데이터 상태 업데이트"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO stock_master (ticker, start_date, end_date, total_rows, last_updated)
            VALUES (?, ?, ?, ?, ?)
        ''', (ticker, start, end, rows, datetime.now()))
        self.conn.commit()

    def get_incomplete_tickers(self, all_tickers, target_rows=1250*381): # 5년치 1분봉 예상치
        """데이터가 부족한 종목 리스트 반환"""
        self.cursor.execute("SELECT ticker FROM stock_master WHERE total_rows < ?", (target_rows,))
        completed = [row[0] for row in self.cursor.fetchall()]
        return list(set(all_tickers) - set(completed))

    def mark_date_data(self, date, data_type, status=1):
        """특정 날짜의 뉴스/거시 데이터 존재 여부 마킹"""
        column = f"has_{data_type}"
        self.cursor.execute(f'''
            INSERT INTO data_availability (date, {column}) VALUES (?, ?)
            ON CONFLICT(date) DO UPDATE SET {column} = ?
        ''', (date, status, status))
        self.conn.commit()
