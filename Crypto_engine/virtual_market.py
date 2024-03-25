"""업비트 거래소의 과거 거래 정보를 이용한 가상 거래소"""
import copy
import requests
from date_converter import DateConverter
from log_manager import LogManager

"""
1. **거래 정보를 입력 받아서 운영한다.**
    1. 거래 정보를 실제 거래소에서 가져온다
    2. 거래 정보 양만큼의 횟수만큼 거래가 가능하다
2. **수수료 비율을 설정할 수 있다**
    1. 기본 수수료 비율을 0.05로 설정하였다
3. **거래 요청을 받아서 가상의 거래 체결 결과를 생성할 수 있다**
    1. 보유 자산 상황을 반영하여 거래 요청 정보에 따른 체결량을 결정한다
    2. 실세 거래 정보를 바탕으로 체결량과 가격을 산출한다
    3. 수수료를 적용한 결과를 생성한다
4. **현재 자산, 보유 종목 정보를 조회할 수 있다**
    1. 거래와 자산의 입출금에 따라 자산, 보유 종목의 내역을 저장해야 한다
5. **아무 거래 없이 다음턴으로 넘어갈 수 있다**
    1. 거래 금액 또는 가격이 0일 경우 해당 턴은 넘어간다
"""


class VirtualMarket:
    """
    거래 요청 정보를 받아서 처리하여 가상의 거래 결과 정보를 생성한다

    """

    URL = "https://api.upbit.com/v1/candles/minutes/1" # simulation_data_provider에도 동일한 코드 존재 
    QUERY_STRING = {"market": "KRW-BTC", "to": "2020-04-30 00:00:00"}

    # URL = "https://api.upbit.com/v1/candles/minutes/1" # virtual market에도 동일한 코드 존재  
    # QUERY_STRING = {"market": "KRW-BTC"}             # 나중에 밖으로 빼서 변수로 받을 것!

    def __init__(self):
        self.logger = LogManager.get_logger(__class__.__name__)
        self.is_initialized = False
        self.data = None # 사용될 거래 정보 목록 
        self.turn_count = 0 # 현재까지 진행된 턴수
        self.balance = 0 # 잔고
        self.commission_ratio = 0.0005 # 수수료율
        self.asset = {} # 자산 목록, 마켓이름을 키값으로 갖고 (평균 매입가격, 수량)을 갖는 딕셔너리 "KRW-BTC":(평균 매입가, 수량)

    def initialize(self, end=None, count=100, budget=0):
        """
        실제 거래소(API)에서 거래 데이터를 가져와서 초기화한다 => simulation_data_provider와 유사한 구조임

        end: 언제까지의 거래기간 정보를 사용할 것인지에 대한 날짜 시간 정보
        count: 거래기간까지 가져올 데이터의 갯수
        """
        if self.is_initialized:
            return

        query_string = copy.deepcopy(self.QUERY_STRING)
        if end is not None:
            query_string["to"] = DateConverter.from_kst_to_utc_str(end) + "Z"

        query_string["count"] = count # 한번에 가져올 데이터의 양 : len(self.data) -> count  

        try:
            response = requests.get(self.URL, params=query_string) # {'market': 'KRW-BTC', 'to': '2020-04-29T22:30:00Z', 'count': 50}
            response.raise_for_status() #  메서드는 응답 객체의 상태 코드를 검사하고, 상태 코드가 400 또는 500 범위 내에 있는 경우에는 예외를 발생시킴. 
                                        # 이렇게 하면 코드에서 HTTP 오류를 처리할 수 있고, 오류가 발생했을 때 적절한 조치를 취할 수 있음
            self.data = response.json()
            self.data.reverse()
            self.balance = budget
            self.is_initialized = True
            self.logger.debug(f"Virtual Market is initialized end: {end}, count: {count}")
        except ValueError as err:
            self.logger.error("Invalid data from server")
            raise UserWarning("fail to get data from server") from err
        except requests.exceptions.HTTPError as err:
            self.logger.error(err)
            raise UserWarning("fail to get data from server") from err
        except requests.exceptions.RequestException as err:
            self.logger.error(err)
            raise UserWarning("fail to get data from server") from err

    def get_balance(self):
        """
        현금을 포함한 모든 자산 정보를 제공한다

        returns:
        {
            balance: 계좌 현금 잔고
            asset: 자산 목록, 마켓이름을 키값으로 갖고 (평균 매입 가격, 수량)을 갖는 딕셔너리
            quote: 종목별 현재 가격 딕셔너리
            date_time: 기준 데이터 시간
        }
        """
        asset_info = {"balance": self.balance}
        quote = None
        try:
            quote = {self.data[self.turn_count]["market"]: self.data[self.turn_count]["trade_price"]}
            
            for name, item in self.asset.items(): # self.asset = {'KRW-BTC': (11372000.0, 0.0009)}
                self.logger.debug(f"asset item: {name}, item price: {item[0]}, amount: {item[1]}")
        except (KeyError, IndexError) as msg:
            self.logger.error(f"invalid trading data {msg}")
            return None

        asset_info["asset"] = self.asset
        asset_info["quote"] = quote
        asset_info["date_time"] = self.data[self.turn_count]["candle_date_time_kst"]
        return asset_info

    def handle_request(self, request):
        """
        거래 요청을 처리해서 결과를 반환
        -> 거래 요청에 대해서 가격, 수량이 현재 가용가능한 자산 상황과 맞는지 비교해보고 buy or sell request로 전달해줌  
        request: 거래 요청 정보
        i.e. {'id': 'request_1', 
              'type': 'buy', 
              'price': 11372000.0, 
              'amount': 0.0009, 
              'date_time': '2020-04-30T14:40:00'}
        
        result:
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
            self.logger.error("virtual market is NOT initialized")
            return None
        now = self.data[self.turn_count]["candle_date_time_kst"] 
        self.turn_count += 1
        next_index = self.turn_count

        if next_index >= len(self.data) - 1: # 가져온 데이터(count=100)의 마지막일경우(next_index=100)
            return {
                "request": request,
                "type": request["type"],
                "price": 0,
                "amount": 0,
                "balance": self.balance,
                "msg": "game-over",
                "date_time": now,
                "state": "done",
            }

        if request["price"] == 0 or request["amount"] == 0:
            self.logger.warning("turn over")
            return "error!"

        if request["type"] == "buy":
            result = self.__handle_buy_request(request, next_index, now)
        elif request["type"] == "sell":
            result = self.__handle_sell_request(request, next_index, now)
        else:
            self.logger.warning("invalid type request")
            result = "error!"
        return result

    def __handle_buy_request(self, request, next_index, dt):
        buy_value = request["price"] * request["amount"]
        buy_total_value = buy_value * (1 + self.commission_ratio)
        old_balance = self.balance

        if buy_total_value > self.balance:
            self.logger.info("no money")
            return "error!"

        try:
            if request["price"] < self.data[next_index]["low_price"]: # 과거 데이터를 이용한 시뮬레이션이므로 ohlc에서 low보다 낮은 주문을 내면 체결이 안되므로 에러를 발생!
                self.logger.info("not matched")
                return "error!"

            name = self.data[next_index]["market"] # {'market': 'KRW-BTC', ...}
            if name in self.asset: # 구매하고자 하는 자산을 과거에 구매하여 보유하고 있는 내역이 있는 경우
                asset = self.asset[name]
                new_amount = asset[1] + request["amount"]
                new_amount = round(new_amount, 6)
                new_value = (request["amount"] * request["price"]) + (asset[0] * asset[1])
                self.asset[name] = (round(new_value / new_amount), new_amount)
            else:
                self.asset[name] = (request["price"], request["amount"]) # {'KRW-BTC': (11372000.0, 0.0009)}

            self.balance -= buy_total_value
            self.balance = round(self.balance)
            self.__print_balance_info("buy", old_balance, self.balance, buy_value) # loger에 구매 내역 저장
            return {
                    "request": request,
                    "type": request["type"],
                    "price": request["price"],
                    "amount": request["amount"],
                    "msg": "success",
                    "balance": self.balance,
                    "state": "done",
                    "date_time": dt,
                    }
        
        except KeyError as msg:
            self.logger.warning(f"internal error {msg}")
            return "error!"

    def __handle_sell_request(self, request, next_index, dt):
        old_balance = self.balance
        try:
            name = self.data[next_index]["market"]
            if name not in self.asset:
                self.logger.info("asset empty")
                return "error!"

            if request["price"] >= self.data[next_index]["high_price"]:
                self.logger.info("not matched")
                return "error!"

            sell_amount = request["amount"]
            if request["amount"] > self.asset[name][1]:
                sell_amount = self.asset[name][1]
                self.logger.warning(
                    f"sell request is bigger than asset {request['amount']} > {sell_amount}"
                )
                del self.asset[name]
            else:
                new_amount = self.asset[name][1] - sell_amount
                new_amount = round(new_amount, 6)
                self.asset[name] = (
                    self.asset[name][0],
                    new_amount,
                )

            sell_value = sell_amount * request["price"]
            self.balance += sell_amount * request["price"] * (1 - self.commission_ratio)
            self.balance = round(self.balance)
            self.__print_balance_info("sell", old_balance, self.balance, sell_value)
            return {
                "request": request,
                "type": request["type"],
                "price": request["price"],
                "amount": sell_amount,
                "msg": "success",
                "balance": self.balance,
                "state": "done",
                "date_time": dt,
            }
        except KeyError as msg:
            self.logger.error(f"invalid trading data {msg}")
            return "error!"

    def __print_balance_info(self, trading_type, old, new, total_asset_value):
        self.logger.debug(f"[Balance] from {old}")
        if trading_type == "buy":
            self.logger.debug(f"[Balance] - {trading_type}_asset_value {total_asset_value}")
        elif trading_type == "sell":
            self.logger.debug(f"[Balance] + {trading_type}_asset_value {total_asset_value}")
        self.logger.debug(f"[Balance] - commission {total_asset_value * self.commission_ratio}")
        self.logger.debug(f"[Balance] to {new}")