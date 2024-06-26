import sqlite3
from log_manager import LogManager
from datetime import datetime

class Database:
    """
    과거 거래 데이터의 데이터 베이스 클래스
    테이블을 생성하고 생성된 테이블 내의 데이터를 조회
    1. 중복 데이터에 대해서 치환함
    2. 중복 되지 않는 새로운 데이터는 데이터 베이스에 추가함 
    
    """

    def __init__(self, db_file=None):
        db = db_file if db_file is not None else "Engine.db"
        self.logger = LogManager.get_logger(__class__.__name__)
        self.conn = sqlite3.connect(db, check_same_thread=False, timeout=30.0) # SQLite 데이터베이스 연결 
        # 기본적으로 check_same_thread는 True며, 만들고 있는 스레드 만 이 연결을 사용할 수 있음. False로 설정하면 반환된 연결을 여러 스레드에서 공유할 수 있음
        # 데이터베이스가 여러 연결을 통해 액세스 되고, 프로세스 중 하나가 데이터베이스를 수정할 때, 해당 트랜잭션이 커밋될 때까지 SQLite 데이터베이스가 잠김. 
        # timeout 매개 변수는 예외를 일으키기 전에 잠금이 해제되기를 연결이 기다려야 하는 시간을 지정함. timeout 매개 변수의 기본값은 5.0(5초)입니다.
        def dict_factory(cursor, row):
            '''쿼리의 결괏값을 키와 딕셔너리형태로 받는 함수'''
            dictionay = {}
            for idx, col in enumerate(cursor.description):
                dictionay[col[0]] = row[idx]
            return dictionay

        self.conn.row_factory = dict_factory

        self.cursor = self.conn.cursor() # SQL 쿼리를 실행하고 결과를 가져오는 작업 수행 (마우스 커서와 같은 기능)
        self.create_table() # 데이터베이스에 저장될 데이터테이블 

    def __del__(self):
        self.conn.close()

    def create_table(self):
        self._create_upbit_table()
        self._create_binance_table()

    def _create_upbit_table(self):
        """테이블 생성
        id TEXT 고유 식별자 period(S)-date_time e.g. 60S-YYYY-MM-DD HH:MM:SS
        period INT 캔들의 기간(초), 분봉 - 60
        recoverd INT 복구된 데이터인지여부
        market TEXT 거래 시장 종류 BTC
        date_time DATETIME 정보의 기준 시간, 'YYYY-MM-DD HH:MM:SS' 형식의 sql datetime format
        opening_price FLOAT 시작 거래 가격
        high_price FLOAT 최고 거래 가격
        low_price FLOAT 최저 거래 가격
        closing_price FLOAT 마지막 거래 가격
        acc_price FLOAT 단위 시간내 누적 거래 금액
        acc_volume FLOAT 단위 시간내 누적 거래 양
        """
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS upbit (id TEXT PRIMARY KEY, period INT, recovered INT, market TEXT, date_time DATETIME \
                , opening_price FLOAT, high_price FLOAT, low_price FLOAT, closing_price FLOAT, acc_price FLOAT, acc_volume FLOAT)"""
        )
        self.conn.commit()

    def _create_binance_table(self):
        """테이블 생성
        id TEXT 고유 식별자 period(S)-date_time e.g. 60S-YYYY-MM-DD HH:MM:SS
        period INT 캔들의 기간(초), 분봉 - 60
        recovered INT 복구된 데이터인지여부
        market TEXT 거래 시장 종류 BTC
        date_time DATETIME 정보의 기준 시간, 'YYYY-MM-DD HH:MM:SS' 형식의 sql datetime format
        opening_price FLOAT 시작 거래 가격
        high_price FLOAT 최고 거래 가격
        low_price FLOAT 최저 거래 가격
        closing_price FLOAT 마지막 거래 가격
        acc_price FLOAT 단위 시간내 누적 거래 금액
        acc_volume FLOAT 단위 시간내 누적 거래 양
        """
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS binance (id TEXT PRIMARY KEY, period INT, recovered INT, market TEXT, date_time DATETIME \
                , opening_price FLOAT, high_price FLOAT, low_price FLOAT, closing_price FLOAT, acc_price FLOAT, acc_volume FLOAT)"""
        )
        self.conn.commit()

    def query(self, start, end, market, period=60, is_upbit=True):
        """데이터 조회"""
        table = "upbit" if is_upbit is True else "binance"

        self.cursor.execute(
            f"SELECT id, period, recovered, market, date_time, opening_price, high_price, low_price, closing_price, \
            acc_price, acc_volume FROM {table} WHERE market = ? AND period = ? AND date_time >= ? AND date_time < ? ORDER BY datetime(date_time) ASC",
            (market, period, start, end),
        )
        return self.cursor.fetchall()

    def update(self, data, period=60, is_upbit=True): # period=60 -> 60초 
        """
        데이터베이스 데이터 추가 또는 업데이트
        - REPLACE는 KEY를 기준으로 동일한 데이터가 존재하면 해 당 데이터를 현재 데이터로 치환함 
        => 만약 동일한 KEY를 갖는 데이터가 없다면 새롭게 생성됨
        """
        table = "upbit" if is_upbit is True else "binance"
        tuple_list = []
        for item in data:
            recovered = item["recovered"] if "recovered" in item else 0 # 입력 안되면 0임 
            tuple_list.append(
                (
                    f"{period}S-{item['date_time']}",
                    period,
                    recovered,
                    item["market"],
                    item["date_time"],
                    item["opening_price"],
                    item["high_price"],
                    item["low_price"],
                    item["closing_price"],
                    item["acc_price"],
                    item["acc_volume"],
                )
            )

        self.logger.info(f"Updated: {len(tuple_list)}")
        self.cursor.executemany( # REPLACE 문 하나로 데이터가 중복 될 경우 데이터의 최신 업데이트 또는 새로운 데이터의 추가가 가능함 
            f"REPLACE INTO {table} (id, period, recovered, market, date_time, opening_price, high_price, low_price, closing_price \
                , acc_price, acc_volume) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple_list,
        )
        self.conn.commit() # DB에 transaction을 commit

if __name__ == '__main__' :
    
    from datetime import datetime
    import os 
    os.chdir('../')
    
    db = Database('Engine.db')
    
    data = [{'period': 60,
              'recovered': 0,
              'market': 'KRW-BTC',
              'date_time': '2020-02-10 03:20:00',
              'opening_price': 11708000.0,
              'high_price': 11721000.0,
              'low_price': 11708000.0,
              'closing_price': 11720000.0,
              'acc_price': 1746637.83688,
              'acc_volume': 0.14911611},
            
             {'period': 60,
              'recovered': 0,
              'market': 'KRW-BTC',
              'date_time': '2020-02-10 03:21:00',
              'opening_price': 11720000.0,
              'high_price': 11721000.0,
              'low_price': 11708000.0,
              'closing_price': 11708000.0,
              'acc_price': 528176.16304,
              'acc_volume': 0.04507455},
             
             {'period': 60,
              'recovered': 0,
              'market': 'KRW-BTC',
              'date_time': '2020-02-10 03:22:00',
              'opening_price': 11709000.0,
              'high_price': 11709000.0,
              'low_price': 11709000.0,
              'closing_price': 11709000.0,
              'acc_price': 8453423.66841,
              'acc_volume': 0.72195949}]
    
    db.update(data)

    start = "2020-02-10T03:20:00"
    end = "2020-02-10T03:23:00"
    # end = "2023-02-10T03:23:00"
    market = "KRW-BTC"
    
    start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
    end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")
    
    # start,end를 기준 db 안의 데이터 조회 가능 
    result = db.query(start_dt, end_dt, market) 
    len(result)
    


