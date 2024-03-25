"""
시뮬레이션을 위한 DataProvider 구현체
Data_provider를 상속받아서 구현함 

"""

import copy
import requests
from date_converter import DateConverter
from data_provider import DataProvider
from log_manager import LogManager


class SimulationDataProvider(DataProvider):
    """
    거래소로부터 과거 데이터를 수집해서 순차적으로 제공하는 클래스

    업비트의 open api를 사용. 별도의 가입, 인증, token 없이 사용 가능
    https://docs.upbit.com/reference#%EC%8B%9C%EC%84%B8-%EC%BA%94%EB%93%A4-%EC%A1%B0%ED%9A%8C
    """

    URL = "https://api.upbit.com/v1/candles/minutes/1" # virtual market에도 동일한 코드 존재  
    QUERY_STRING = {"market": "KRW-BTC"}             # 나중에 밖으로 빼서 변수로 받을 것!

    def __init__(self):
        self.logger = LogManager.get_logger(__class__.__name__)
        self.is_initialized = False
        self.data = []
        self.index = 0 # 순차적 데이터 제공을 위한 index

    def initialize_simulation(self, end=None, count=100):
        """Open Api를 사용해서 데이터를 가져와서 초기화한다"""

        self.index = 0
        query_string = copy.deepcopy(self.QUERY_STRING)

        try:
            if end is not None:
                query_string["to"] = DateConverter.from_kst_to_utc_str(end) + "Z"
            query_string["count"] = count

            response = requests.get(self.URL, params=query_string)
            response.raise_for_status() # request에 문제가 있다면 에러 발생 
            self.data = response.json()
            self.data.reverse()
            self.is_initialized = True
            self.logger.info(f"data is updated from server # end: {end}, count: {count}")
            
        except ValueError as error:
            self.logger.error("Invalid data from server")
            raise UserWarning("Fail get data from sever") from error
        except requests.exceptions.HTTPError as error:
            self.logger.error(error)
            raise UserWarning("Fail get data from sever") from error
        except requests.exceptions.RequestException as error:
            self.logger.error(error)
            raise UserWarning("Fail get data from sever") from error

    def get_info(self):
        """순차적으로 거래 정보 전달한다

        Returns: 거래 정보 딕셔너리
        {
            "market": 거래 시장 종류 BTC
            "date_time": 정보의 기준 시간
            "opening_price": 시작 거래 가격
            "high_price": 최고 거래 가격
            "low_price": 최저 거래 가격
            "closing_price": 마지막 거래 가격
            "acc_price": 단위 시간내 누적 거래 금액
            "acc_volume": 단위 시간내 누적 거래 양
        }
        """
        now = self.index # 현재 순번을 index에 저장하고 get_info가 호출 될 때마다 다음 데이터를 전달하도록 함. 
        # 실제 시간이 지나서 정보를 요청하는 것처럼 처리하기 위해 요청에 따라 그 다음의 정보를 순차적으로 제공
        # 즉, initialize_simulation에서 count=100 만큼 데이터를 한번에 가져 오지만 제공할 때는 self.index 수를 하나씩 늘려가며 순차적으로 제공
        if now >= len(self.data):
            return None

        self.index = now + 1
        self.logger.info(f'[DATA] @ {self.data[now]["candle_date_time_kst"]}')
        return self.__create_candle_info(self.data[now]) #index 순서대로 하나씩 데이터 제공

    def __create_candle_info(self, data):
        try:
            return {
                "market": data["market"],
                "date_time": data["candle_date_time_kst"],
                "opening_price": data["opening_price"],
                "high_price": data["high_price"],
                "low_price": data["low_price"],
                "closing_price": data["trade_price"],
                "acc_price": data["candle_acc_trade_price"],
                "acc_volume": data["candle_acc_trade_volume"],
            }
        except KeyError:
            self.logger.warning("invalid data for candle info")
            return None




if __name__ == '__main__':

    dr = SimulationDataProvider()
    dr.initialize_simulation()
    data2 = dr.get_info()

    dp = SimulationDataProvider()
    end_date = "2020-04-30T07:30:00"
    dp.initialize_simulation(end=end_date, count=50)
    
    # 첫번째 데이터 
    print(dp.get_info())
    
    # 두번째 데이터 
    print(dp.get_info())





