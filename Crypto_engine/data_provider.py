"""
거래에 관련된 데이터를 수집해서 정보를 제공

이 모듈은 거래에 관련된 데이터를 수집해서 필요한 데이터 포맷에 맞게 정보를 제공하는 클래스인 DataProvider 추상클래스다.

구현해야할 SimulationDataProvider 클래스는 DataProvider라는 추상 클래스를 상속받아서 클래스를 만들었다. 
DataProvider를 import하고 클래스를 선언할 때 파라미터로 상속받을 클래스 DataProvider를 넣어주면 된다. 

"""

from abc import ABCMeta, abstractmethod


class DataProvider(metaclass=ABCMeta):
    """
    데이터 소스로부터 데이터를 수집해서 정보를 제공하는 클래스
    """

    @abstractmethod
    def get_info(self):
        """
        현재 거래 정보를 딕셔너리로 전달

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



