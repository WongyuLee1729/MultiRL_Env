"""분할 매수 후 홀딩 하는 간단한 전략"""

import copy
import time
from datetime import datetime
from strategy import Strategy
from log_manager import LogManager


class StrategyBuyAndHold(Strategy):
    """
    분할 매수 후 홀딩 하는 간단한 전략

    is_Initialized: 최초 잔고는 초기화 할 때만 갱신 된다
    data: 거래 데이터 리스트, OHLCV 데이터
    result: 거래 요청 결과 리스트
    request: 마지막 거래 요청
    budget: 시작 잔고
    balance: 현재 잔고
    min_price: 최소 주문 금액
    """

    ISO_DATEFORMAT = "%Y-%m-%dT%H:%M:%S"
    COMMISSION_RATIO = 0.0005

    def __init__(self):
        self.is_intialized = False
        self.is_simulation = False # ? => simulation과 replay 초기설정은 False
        self.data = []
        self.budget = 0
        self.balance = 0.0
        self.min_price = 0
        self.result = []
        self.request = None
        self.logger = LogManager.get_logger(__class__.__name__)
        self.name = "BnH"
        self.waiting_requests = {} # 1.요청해서 체결된 거래와 2.요청되었지만 체결되지 않은 거래가 혼재하면 1번을 제거하기 위한 변수 
            

    def initialize(self, budget, min_price=5000):
        """
        예산과 최소 거래 가능 금액을 설정한다
        """
        if self.is_intialized:
            return

        self.is_intialized = True
        self.budget = budget
        self.balance = budget
        self.min_price = min_price


    def update_trading_info(self, info): 
        """새로운 거래 정보를 업데이트

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
        if self.is_intialized is not True:
            return
        self.data.append(copy.deepcopy(info))


    def get_request(self):
        """
        데이터 분석 결과에 따라 거래 요청 정보를 생성한다

        5번에 걸쳐 분할 매수 후 홀딩하는 전략   --> 20번으로 변경!
        마지막 종가로 처음 예산의 1/5에 해당하는 양 만큼 매수시도
        Returns: 배열에 한 개 이상의 요청 정보를 전달
        [{
            "id": 요청 정보 id "1607862457.560075"
            "type": 거래 유형 sell, buy, cancel
            "price": 거래 가격
            "amount": 거래 수량
            "date_time": 요청 데이터 생성 시간, 시뮬레이션 모드에서는 데이터 시간
        }]
        """
        if self.is_intialized is not True:
            return None

        try:
            if len(self.data) == 0 or self.data[-1] is None:
                raise UserWarning("data is empty")

            last_closing_price = self.data[-1]["closing_price"]
            now = datetime.now().strftime(self.ISO_DATEFORMAT)

            if self.is_simulation:
                now = self.data[-1]["date_time"]

            target_budget = self.budget / 20 
            if target_budget > self.balance:
                target_budget = self.balance

            amount = round(target_budget / last_closing_price, 6)
            trading_request = {
                                "id": str(round(time.time(), 3)),
                                "type": "buy",
                                "price": last_closing_price,
                                "amount": amount,
                                "date_time": now,
                                }
            total_value = round(float(last_closing_price) * amount)

            if self.min_price > total_value or total_value > self.balance:
                raise UserWarning("total_value or balance is too small")

            self.logger.info(f"[REQ] id: {trading_request['id']} =====================")
            self.logger.info(f"price: {last_closing_price}, amount: {amount}")
            self.logger.info(f"type: buy, total value: {total_value}")
            self.logger.info("================================================")
            final_requests = []
            for request_id in self.waiting_requests:
                self.logger.info(f"cancel request added! {request_id}")
                # final_requests: waiting_requests에 체결되지 않고 대기중인 거래 요청이 있으면 
                # 최소 요청을 추가하고 
                final_requests.append(
                    {
                        "id": request_id,
                        "type": "cancel",
                        "price": 0,
                        "amount": 0,
                        "date_time": now,
                    }
                )
            final_requests.append(trading_request)
            return final_requests
        
        except (ValueError, KeyError) as msg:
            self.logger.error(f"invalid data {msg}")
        except IndexError:
            self.logger.error("empty data")
        except AttributeError as msg:
            self.logger.error(msg)
        except UserWarning as msg:
            self.logger.info(msg)
            if self.is_simulation:
                return [
                    {
                        "id": str(round(time.time(), 3)),
                        "type": "buy",
                        "price": 0,
                        "amount": 0,
                        "date_time": now,
                    }
                ]
            return None



    def update_result(self, result):
        """
        요청한 거래의 결과를 업데이트

        request: 거래 요청 정보
        result:
        {
            "request": 요청 정보
            "type": 거래 유형 sell, buy, cancel
            "price": 거래 가격
            "amount": 거래 수량
            "msg": 거래 결과 메세지
            "state": 거래 상태 requested, done
            "date_time": 시뮬레이션 모드에서는 데이터 시간 +2초
        }
        """
        if self.is_intialized is not True:
            return

        try:
            request = result["request"]
            if result["state"] == "requested":
                self.waiting_requests[request["id"]] = result
                return

            if result["state"] == "done" and request["id"] in self.waiting_requests:
                del self.waiting_requests[request["id"]]

            total = float(result["price"]) * float(result["amount"])
            fee = total * self.COMMISSION_RATIO
            if result["type"] == "buy":
                self.balance -= round(total + fee)
            else:
                self.balance += round(total - fee)

            self.logger.info(f"[RESULT] id: {result['request']['id']} ================")
            self.logger.info(f"type: {result['type']}, msg: {result['msg']}")
            self.logger.info(f"price: {result['price']}, amount: {result['amount']}")
            self.logger.info(f"total: {total}, balance: {self.balance}")
            self.logger.info("================================================")
            self.result.append(copy.deepcopy(result))
        except (AttributeError, TypeError) as msg:
            self.logger.error(msg)
            

if __name__ == '__main__':
    '''
    1. intialize 메서드를 통해 초기화한다
    2. update_trading_info 메서드를 통해서 거래 정보를 전달 받는다. 
    3. get_request 메서드가 호출되면 업데이트 된 정보를 바탕으로 매매 요청 정보를 반환한다
    4. update_result 메서드를 통해서 거래 결과를 업데이트 받는다
    '''
    
    strategy = StrategyBuyAndHold()
    # strategy.get_request() #-> 초기화 되지 않은 상태로 get_request를 실행하면 결과값이 없음
    
    strategy.initialize(500000, 5000) # 초기화 - 예산 및 최소 주문 금액 입력 
    
    # 거래 정보를 입력 -> self.data에 append 시킴 
    # 원래는 data provider 에서 전달 받음
    strategy.update_trading_info(
        {
            "market": "KRW-BTC",
            "date_time": "2020-04-30T14:51:00",
            "opening_price": 11288000.0,
            "high_price": 11304000.0,
            "low_price": 11282000.0,
            "closing_price": 11304000.0,
            "acc_price": 587101574.8949,
            "acc_volume": 51.97606868,
        }
    )
    
    print(strategy.data[-1]) # 입력된 거래 정보를 확인하고 싶다면 맴버 변수를 출력해서 볼 수 있음 
    # 거래 요청 정보가 마지막 거래 정보의 closing_price 기준으로 buy 정보로 생성되는지 확인 
    # 거래 대금 총량이 초기 예산의 1/5 분량의 amount인지 확인 
    # requested 상태의 거래가 있을 경우 첫번째 거래 요청은 cancel 요청인지 확인 
    
    # 거래 요청 정보 생성
    expected_request = {
        "type": "buy",
        "price": 11304000.0,
        "amount": 0.0009,
    }
    request = strategy.get_request() # 하나 이상의 요청을 할 수 있음 
    last_id = request[0]["id"] 
    print(request)

    # 앞서 생성된 거래 요청 정보의 결과를 생성해서 업데이트 했을때,
    # 입력된 거래 결과에 따라 balance가 제대로 업데이트 되는지 확인

    # 거래 결과 입력 - 정상 체결 됨 (type buy or sell)
    strategy.update_result({  # 거래 결과 -> virtual market으로부터 받음 : 거래 요청과 결과로 이루어진 dictionary
                                "request": {
                                            "id": last_id,
                                            "type": "buy",
                                            "price": 11304000.0,
                                            "amount": 0.0009,
                                            "date_time": "2020-04-30T14:51:00",
                                            },
                                
                                "type": "buy",
                                "price": 11304000.0,
                                "amount": 0.0009,
                                "msg": "success",
                                "balance": 0,
                                "state": "done",
                                "date_time": "2020-04-30T14:51:00",
                          })

    print(strategy.balance) # 39821
    print(strategy.result)
    
    







