"""

데이터를 기반으로 매매 결정을 생성하는 Strategy 추상클래스

1. 초기 자산을 설정할 수 있다 (Simulation operator에서 받음)

2. 데이터를 입력 받아서 저장한다 (Data Provider에서 받음)
 - 제공되는 데이터를 입력 받을 때마다 모두 저장하여 이후 분석에 사용한다 
 
3. 입력 받은 데이터와 매매 결과를 바탕으로 전략에 따른 거래 요청 정보를 생성한다 (Trader, virtual market에 전략의 대한 거래를 요청한다)
 - 데이터와 결과, 자산을 바탕으로 전략에 따른 거래 요청 정보를 생성한다
 
 3.1 거래 요청 정보는 1개만 유지한다. 
 - 이전에 생성한 거래 요청 정보가 완료되지 않았을 경우 새로운 거래 요청 정보를 생성할 때 
   이전 거래 요청 정보의 취소 요청 정보도 함께 생성하여 전달한다
   
4. 거래 요청의 결과를 저장한다 (Trader, virtual market으로부터 전략으로 인한 거래 결과를 반환 받아 저장한다)
 - 거래 요청 결과를 모두 저장하여 이후 분석에 사용한다

 4.1 거래 요청의 결과 정보를 통해 자산 현황을 갱신한다
 - 전략에 반영되는 자산 현황은 거래 요청 결과를 통해 갱신한다

"""


'''
algo-engine에서는 strategy를 비롯해서 dataprovider, trader 등이 거래소나 전략에 따라 자유롭게 조합하여 동작시킬 수 있다. 
사용되는 모듈이 다른 모듈로 교체되어도 관련된 다른 모듈에 영향없이 프로그램이 정상 동작하기 위해서는 각 모듈이 어떻게 동작할지 사전에 약속을 정하고, 
그 약속을 잘 따라야 한다 그 약속이 인터페이스이고, 추상 클래스의 메서드인 것이다. 따라서 메서드를 정할 때 이름부터 파라미터, 역할까지 신중하게 정하고 그에 맞게 모듈을 구현해야 한다. 

Strategy 추상 클래스에는 4개의 추상 메서드를 선언했다. 예산을 설정할 수 있는 initialize 메서드와 전략에 따라서 거래 요청 정보를 생성하는 get_request 메서드를 만들었다. 
거래 요청 정보는 가격과 수량, 거래 유형에 대한 정보로써 주어진 데이터를 바탕으로 전략이 생성해서 제공하는 핵심 정보이다. 

새로운 거래 정보를 업데이트하는 update_trading_info 메서드와 거래 요청에 따라 거래가 체결되었을 경우 결과를 입력 받을 수 있는 update_result 메서드도 추가했다. 
'''


from abc import ABCMeta, abstractmethod


class Strategy(metaclass=ABCMeta):
    """
    데이터를 받아서 매매 판단을 하고 결과를 받아서 다음 판단에 반영하는 전략 클래스
    """

    @abstractmethod
    def initialize(self, budget, min_price=100):
        """예산을 설정하고 초기화한다"""

    @abstractmethod
    def get_request(self):
        """
        전략을 만들고 해 당 전략에 따라 거래 요청 정보를 생성한다

        Returns: 배열에 한 개 이상의 요청 정보를 전달
        [{
            "id": 요청 정보 id "1607862457.560075"
            "type": 거래 유형 sell, buy, cancel
            "price": 거래 가격
            "amount": 거래 수량
            "date_time": 요청 데이터 생성 시간, 시뮬레이션 모드에서는 데이터 시간
        }]
        """

    @abstractmethod
    def update_trading_info(self, info):
        """
        새로운 거래 정보를 업데이트

        info:
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

    @abstractmethod
    def update_result(self, result):
        """요청한 거래의 결과를 업데이트
        request: 거래 요청 정보
        result:
        {
            "request": 요청 정보
            "type": 거래 유형 sell, buy, cancel
            "price": 거래 가격
            "amount": 거래 수량
            "state": 거래 상태 requested, done
            "msg": 거래 결과 메세지
            "date_time": 거래 체결 시간, 시뮬레이션 모드에서는 데이터 시간 +2초
        }
        """