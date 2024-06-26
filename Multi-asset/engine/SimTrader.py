"""시뮬레이션을 위한 가상 거래를 처리"""

from log_manager import LogManager
from trader import Trader
from VirMarket import VirtualMarket


class SimulationTrader(Trader):
    """
    거래 요청 정보를 받아서 거래소에 요청하고 거래소에서 받은 결과를 제공해주는 클래스

    id: 요청 정보 id "1607862457.560075"
    type: 거래 유형 sell, buy, cancel
    price: 거래 가격
    amount: 거래 수량
    """

    def __init__(self):
        self.logger = LogManager.get_logger(__class__.__name__)
        self.market = VirtualMarket()
        self.is_initialized = False
        self.name = "Simulation"

    def initialize_simulation(self, end, count, budget):
        """시뮬레이션기간, 횟수, 예산을 초기화 한다"""
        self.market.initialize(end, count, budget) # virtual market 모듈에서 실제 거래소 API를 통해 필요한 데이터를 요청하고 초기화함
        self.is_initialized = True

    def send_request(self, request_list, callback):
        """
        거래 요청을 처리한다

        1. strategy의 결과로 받은 요청 정보를 virtual_market에 보내주고 거래를 요청하고, 
        2. virtual_market으로부터 요청에 대한 거래 결과를 받아 콜백 함수를 통해 결과를 넘겨 줌 
        request_list: 한 개 이상의 거래 요청 정보 리스트 -> strategy를 거쳐 나온 거래 요청 내용
        
            [{
            "id": 요청 정보 id "1607862457.560075"
            "type": 거래 유형 sell, buy, cancel
            "price": 거래 가격
            "amount": 거래 수량
            "date_time": 요청 데이터 생성 시간
            }]
        
        callback(result): -> callback 함수만 받고 result는 virtual_market에서 따로 받아 넣어 줌
            { "request": 요청 정보 전체
             "type": 거래 유형 sell, buy, cancel
             "price": 거래 가격
             "amount": 거래 수량
             "msg": 거래 결과 메세지 success, internal error
             "balance": 거래 후 계좌 현금 잔고
             "state": 거래 상태 requested, done
             "date_time": 거래 체결 시간, 시뮬레이션 모드에서는 request의 시간 }
        
        result:  -> from virtual_market
            { "request": 요청 정보
                "type": 거래 유형 sell, buy, cancel
                "price": 거래 가격
                "amount": 거래 수량
                "state": 거래 상태 requested, done
                "msg": 거래 결과 메세지
                "date_time": 시뮬레이션 모드에서는 데이터 시간
                "balance": 거래 후 계좌 현금 잔고 }
        """

        if self.is_initialized is not True:
            raise UserWarning("Not initialzed")

        try:
            result = self.market.handle_request(request_list[0]) # request_list[0]: [0]은 가장 바깥쪽 list를 벗겨내기 위함 [{}] -> {}
            callback(result)
        except (TypeError, AttributeError) as msg:
            self.logger.error(f"invalid state {msg}")
            raise UserWarning("invalid state") from msg

    def get_account_info(self):
        """계좌 요청 정보를 요청한다
        현금을 포함한 모든 자산 정보를 제공한다

        returns:
        {
            balance: 계좌 현금 잔고
            asset: 자산 목록, 마켓이름을 키값으로 갖고 (평균 매입 가격, 수량)을 갖는 딕셔너리
            quote: 종목별 현재 가격 딕셔너리
        }
        """

        if self.is_initialized is not True:
            raise UserWarning("Not initialzed")

        try:
            return self.market.get_balance()
        except (TypeError, AttributeError) as msg:
            self.logger.error(f"invalid state {msg}")
            raise UserWarning("invalid state") from msg

    def cancel_request(self, request_id):
        """거래 요청을 취소한다
        request_id: 취소하고자 하는 request의 id
        """

    def cancel_all_requests(self):
        """모든 거래 요청을 취소한다
        체결되지 않고 대기중인 모든 거래 요청을 취소한다
        """

    
if __name__ == "__main__":
    
    trader = SimulationTrader()
    end_date = "2020-04-30T01:00:00"
    # 시뮬레이션 기간 횟수 예산 초기화-> virtual market의 초기화를 위해 입력값을 전달해주는 정도의 초기화임 
    trader.initialize_simulation(end=end_date, count=50, budget=50000)
    
    # request는 테스트하고 싶은 내용으로 생성해서 요청해 보고 결과를 출력창에서 확인해보자
    request = [ {"id": "request_1",
                "type": "buy",
                "price": 11372000.0,
                "amount": 0.0009,
                "date_time": "2020-04-30T14:40:00",}]
    
    result = None

    def send_request_callback(callback_result): # callback:다른 함수에 인자로 넘겨지는 함수 
        # nonlocal result
        result = callback_result

    # requset = [{'id': 'request_1', 'type': 'buy', 'price': 11372000.0, 'amount': 0.0009, 'date_time': '2020-04-30T14:40:00'}]
    trader.send_request(request, send_request_callback) # 여기서 좀 헷갈리는데 거래는 요청하는 send_request에 거래요청의 결과인 callback을 넣어준다는게 이상함
    # 하지만 자세히 보면 callback으로 들어가는 값은 변수가 아니라 함수임 (=send_request_callback)
    # 따라서 send_request 함수 안에서 거래를 요청하고 요청한 내역을 virtual_market에 보내고, 그 결과를 send_request_callback 함수에 보내 최종적으로 callback 결과를 산출함


    # 현재 계좌 정보를 조회한다 
    trader.get_account_info() # 반환값이 출력창에 출력된다
    
    # Out[4]: 
    # {'balance': 29520,
    #  'asset': {'KRW-BTC': (11372000, 0.0018)},
    #  'quote': {'KRW-BTC': 10024000.0},
    #  'date_time': '2020-04-30T00:12:00'}
    

