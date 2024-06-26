"""거래 요청, 결과 정보를 저장하고 투자 결과를 분석

이 모듈은 거래 요청, 결과 정보를 저장하고 투자 결과를 분석하는 클래스인 Analayzer를 포함하고 있다.
"""
import copy
import os
from datetime import datetime
import matplotlib
import pandas as pd
import mplfinance as mpf
from log_manager import LogManager

matplotlib.use("Agg")


class Analyzer:
    """
    거래 요청, 결과 정보를 저장하고 투자 결과를 분석하는 클래스
    """

    ISO_DATEFORMAT = "%Y-%m-%dT%H:%M:%S"
    OUTPUT_FOLDER = "output/"
    SMA = (5, 20)

    def __init__(self, RECORD_INTERVAL=1, sma=None,rsi=None,macd=None,adx=None):
        self.RECORD_INTERVAL = RECORD_INTERVAL # 자산 정보를 주기적으로 업데이트하는 최소 시간 default 60
        # self.SMA = sma
        self.RSI = rsi
        self.MACD = macd
        self.ADX = adx
        
        self.request_list = [] # 거래 요청 데이터 목록 
        self.result_list = [] # 거래 결과 데이터 목록 
        self.info_list = [] # trading request data info -> def put_trading_info
        self.asset_info_list = [] # 특정 시점에 기록된 자산 데이터 목록 
        self.score_list = [] # 특정 시점에 기록된 수익률 데이터 목록
        self.get_asset_info_func = None # 자산 정보 업데이트를 요청하기 위한 콜백 함수 
        self.logger = LogManager.get_logger(__class__.__name__)
        self.is_simulation = False
        
        
        
        if os.path.isdir("output") is False:
            print("create output folder")
            os.mkdir("output")

    # 기본적으로 analyzer는 정보를 받는 주체일 뿐 정보를 어디어 어떻게 요청할지 알 수 없음 
    # analyzer에 해 당 요청 기능을 넣기 보다는 get_asset_info_func라는 함수를 요청에 대한 인터페이스 함수로 정하고 
    # 요청 받을 수 있는 인터페이스를 통해서 의존성 없이 연동하는 것이 더 좋을 것 같아 해 당 콜백 함수를 이용함
    # => 자산 정보를 생성하는 것과 전달해주는 것에 대해서 Analyzer는 전혀 관여하지 않고 단지 정해진 인터페이스를 통해서 정보를 전달받기 위함 
    
    def initialize(self, get_asset_info_func): 
        """
        콜백 함수를 입력받아 초기화한다

        Args: get_asset_info_func: 거래 데이터를 요청하는 함수로 func(arg1) arg1은 정보 타입

        i.e. arg1 = {"balance": 5000,
                     "asset": {},
                     "quote": {"KRW-BTC": 26100},
                     "date_time": "2020-12-21T01:13:00",}
        """
        self.get_asset_info_func = get_asset_info_func


    def put_requests(self, requests):
        """거래 요청 정보를 저장한다

        request:
        {
            "id": 요청 정보 id "1607862457.560075"
            "type": 거래 유형 sell, buy, cancel
            "price": 거래 가격
            "amount": 거래 수량
            "date_time": 요청 데이터 생성 시간, 시뮬레이션 모드에서는 데이터 시간
        }
        report_type: 보고서를 위한 데이터 종류
            0: 거래 데이터
            1: 매매 요청
            2: 매매 결과
            3: 수익률 정보
        """
        for request in requests:
            new = copy.deepcopy(request)
            if request["type"] == "cancel":
                new["price"] = 0
                new["amount"] = 0
            else:
                if float(request["price"]) <= 0 or float(request["amount"]) <= 0:
                    continue
                new["price"] = float(new["price"])
                new["amount"] = float(new["amount"])
            new["report_type"] = 1
            self.request_list.append(new)

    def put_result(self, result):
        """
        거래 결과 정보를 저장한다

        request: 거래 요청 정보
        result:
        {
            "request": 요청 정보
            "type": 거래 유형 sell, buy, cancel
            "price": 거래 가격
            "amount": 거래 수량
            "state": 거래 상태 requested, done
            "msg": 거래 결과 메세지
            "date_time": 시뮬레이션 모드에서는 데이터 시간 +2초
        }
        report_type: 보고서를 위한 데이터 종류
            0: 거래 데이터
            1: 매매 요청
            2: 매매 결과
            3: 수익률 정보
        """

        try:
            if float(result["price"]) <= 0 or float(result["amount"]) <= 0:
                return
        except KeyError:
            self.logger.warning("Invalid result")
            return

        new = copy.deepcopy(result)
        new["price"] = float(new["price"])
        new["amount"] = float(new["amount"])
        new["report_type"] = 2
        self.result_list.append(new)
        self.update_asset_info() # 자산 정보를 저장

    def put_trading_info(self, info):
        """거래 요청 및 결과에 대한 개괄적인 정보를 저장한다

        report_type: 보고서를 위한 데이터 종류
            0: 거래 데이터
            1: 매매 요청
            2: 매매 결과
            3: 수익률 정보
        """
        new = copy.deepcopy(info)
        new["report_type"] = 0 # create the key named "report_type"
        self.info_list.append(new)
        self.make_periodic_record() # 수익율 기록 -> datetime.now() 날짜 기준 기록


    def update_asset_info(self):
        """
        자산 정보를 저장한다

        returns:
        {
            balance: 계좌 현금 잔고
            asset: 자산 목록, 마켓이름을 키값으로 갖고 (평균 매입 가격, 수량)을 갖는 딕셔너리
            quote: 종목별 현재 가격 딕셔너리
        }
        """
        if self.get_asset_info_func is None:
            self.logger.warning("get_asset_info_func is NOT set")
            return

        asset_info = self.get_asset_info_func()
        new = copy.deepcopy(asset_info)
        new["balance"] = float(new["balance"])
        if self.is_simulation is True and len(self.info_list) > 0:
            new["date_time"] = self.info_list[-1]["date_time"]
        self.asset_info_list.append(new)
        self.make_score_record(new)

    def make_start_point(self):
        """시작시점 거래정보를 기록한다"""
        self.request_list = [] # defined in init but set it empty
        self.result_list = [] # defined in init but set it empty
        self.asset_info_list = [] # defined in init but set it empty
        self.update_asset_info()

    def make_periodic_record(self):
        """주기적으로 수익율을 기록한다"""
        now = datetime.now()
        if self.is_simulation:
            now = datetime.strptime(self.info_list[-1]["date_time"], self.ISO_DATEFORMAT)

        last = datetime.strptime(self.asset_info_list[-1]["date_time"], self.ISO_DATEFORMAT)
        delta = now - last
        # 자산 업데이트 interval = 60 보다 현재 info_list에 마지막으로 기록된 날짜와 asset_info_list 마지막으로 기록된 날의 차이를 비교하여 자산정보를 업데이트함 
        if delta.total_seconds() > self.RECORD_INTERVAL:
            self.update_asset_info()

    def make_score_record(self, new_info):
        """
        수익률 기록을 생성한다

        score_record:
            balance: 계좌 현금 잔고
            cumulative_return: 기준 시점부터 누적 수익률
            price_change_ratio: 기준 시점부터 보유 종목별 가격 변동률 딕셔너리
            asset: 자산 정보 튜플 리스트 (종목, 평균 가격, 현재 가격, 수량, 수익률(소숫점3자리))
            date_time: 데이터 생성 시간, 시뮬레이션 모드에서는 데이터 시간
            report_type: 3, 보고서를 위한 데이터 종류
        """

        try:
            start_total = self.__get_start_property_value() # 시작 시점의 자산 총액 
            start_quote = self.asset_info_list[0]["quote"] # 시작 시점의 시세 정보
            current_total = float(new_info["balance"]) # 현시점의 자산 총액
            current_quote = new_info["quote"] # 현 시점의 시세 정보
            cumulative_return = 0 # 누적 수익률 
            new_asset_list = []
            price_change_ratio = {} # 시세 변동폭 확인을 위한 가격 변동률
            self.logger.debug(f"make_score_record new_info {new_info}")

            for name, item in new_info["asset"].items(): 
                item_yield = 0
                amount = float(item[1])
                buy_avg = float(item[0])
                price = float(current_quote[name])
                current_total += amount * price
                item_price_diff = price - buy_avg
                if item_price_diff != 0 and buy_avg != 0:
                    item_yield = (price - buy_avg) / buy_avg * 100
                    item_yield = round(item_yield, 3)

                self.logger.debug(
                    f"yield record {name}, buy_avg: {buy_avg}, {price}, {amount}, {item_yield}"
                )
                new_asset_list.append((name, buy_avg, price, amount, item_yield))
                start_price = start_quote[name]
                price_change_ratio[name] = 0
                price_diff = price - start_price
                if price_diff != 0:
                    price_change_ratio[name] = price_diff / start_price * 100
                    price_change_ratio[name] = round(price_change_ratio[name], 3)
                self.logger.debug(
                    f"price change ratio {start_price} -> {price}, {price_change_ratio[name]}%"
                )

            total_diff = current_total - start_total
            if total_diff != 0:
                cumulative_return = (current_total - start_total) / start_total * 100
                cumulative_return = round(cumulative_return, 3)
            self.logger.info(
                f"cumulative_return {start_total} -> {current_total}, {cumulative_return}%"
            )

            self.score_list.append(
                {
                    "balance": float(new_info["balance"]),
                    "cumulative_return": cumulative_return,
                    "price_change_ratio": price_change_ratio,
                    "asset": new_asset_list,
                    "date_time": new_info["date_time"],
                    "report_type": 3,
                }
            )
        except (IndexError, AttributeError) as msg:
            self.logger.error(f"making score record fail {msg}")

    def get_return_report(self, graph_filename=None):
        """현시점 기준 간단한 수익률 보고서를 제공한다

        Returns:
            (
                start_budget: 시작 자산
                final_balance: 최종 자산
                cumulative_return : 기준 시점부터 누적 수익률
                price_change_ratio: 기준 시점부터 보유 종목별 가격 변동률 딕셔너리
                graph: 그래프 파일 패스
            )
        """
        self.update_asset_info()
        """
        __get_start_propoert_value 메서드는 자산 정보 리스트 asset_info_list의 첫 번째 값을 참조하여 
        시작 시점의 자산가치를 계산해준다. 마찬가지로 __get_last_property_value는 자산 정보 리스트 asset_info_list의 마지막 값을 참조하여
        현재의 자산가치를 계산해준다. 최종 수익률과 가격 변동률은 수익률 정보 리스트 score_list의 마지막 값을 통해서 얻을 수 있다. 
        이렇게 4개의 값을 튜플로 만들어서 반환한다. 
        """

        try:
            graph = None
            start_value = self.__get_start_property_value()
            last_value = self.__get_last_property_value()
            last_return = self.score_list[-1]["cumulative_return"]
            change_ratio = self.score_list[-1]["price_change_ratio"]
            if graph_filename is not None:
                graph = self.__draw_graph(graph_filename, is_fullpath=True)

            summary = (start_value, last_value, last_return, change_ratio, graph)
            self.logger.info("### Return Report ===============================")
            self.logger.info(f"Property                 {start_value:10} -> {last_value:10}")
            self.logger.info(
                f"Gap                                    {last_value - start_value:10}"
            )
            self.logger.info(f"Cumulative return                    {last_return:10} %")
            self.logger.info(f"Price_change_ratio {change_ratio}")
            return summary
        except (IndexError, AttributeError):
            self.logger.error("get return report FAIL")

    def get_trading_results(self):
        """거래 결과 목록을 반환한다"""
        return self.result_list

    def create_report(self, filename=None, tag=None):
        """수익률 보고서를 생성한다

        수익률 보고서를 생성하고, 그래프를 화면에 출력한다.
        Args:
            filename: 생성할 리포트 파일명
        Returns:
            {
                "summary": (
                    start_budget: 시작 자산
                    final_balance: 최종 자산
                    cumulative_return : 기준 시점부터 누적 수익률
                    price_change_ratio: 기준 시점부터 보유 종목별 가격 변동률 딕셔너리
                    graph: 그래프 파일 패스
                ),
                "trading_table" : [
                    {
                        "date_time": 생성 시간, 정렬 기준 값
                        거래 정보, 매매 요청 및 결과 정보, 수익률 정보 딕셔너리
                    }
                ]
            }
        """
        final_filename = "untitled-report"
        if tag is not None:
            final_filename = tag

        if filename is not None:
            final_filename = filename

        try:
            summary = self.get_return_report()
            if summary is None:
                self.logger.error("invalid return report")
                return None

            list_sum = self.request_list + self.info_list + self.score_list + self.result_list
            trading_table = sorted(
                list_sum,
                key=lambda x: (
                    datetime.strptime(x["date_time"], self.ISO_DATEFORMAT),
                    x["report_type"],
                ),
            )
            self.__create_report_file(final_filename, summary, trading_table)
            self.__draw_graph(final_filename)
            return {"summary": summary, "trading_table": trading_table}
        except (IndexError, AttributeError):
            self.logger.error("create report FAIL")

    def __create_report_file(self, filepath, summary, trading_table):
        """
        보고서를 정해진 형식에 맞게 파일로 출력한다

        ### TRADING TABLE =================================
        date_time, opening_price, high_price, low_price, closing_price, acc_price, acc_volume
        2020-02-23T00:00:00, 5000, 15000, 4500, 5500, 1500000000, 1500

        date_time, [->] id, type, price, amount
        2020-02-23T00:00:01, 1607862457.560075, sell, 1234567890, 1234567890

        date_time, [<-] request id, type, price, amount, msg
        2020-02-23T00:00:01, 1607862457.560075, sell, 1234567890, 1234567890, success, 1234567890

        date_time, [#] balance, cumulative_return, price_change_ratio, asset
        2020-02-23T00:00:01, 1234567890, 1234567890, 1234567890, 1234567890

        ### SUMMARY =======================================
        Property                 1234567890 -> 1234567890
        Gap                                    1234567890
        Cumulative return                    1234567890 %
        Price_change_ratio {'mango': -50.0, 'apple': 50.0}
        """
        final_path = self.OUTPUT_FOLDER + filepath + ".txt"
        with open(final_path, "w") as report_file:
            if len(trading_table) > 0:
                report_file.write("### TRADING TABLE =================================\n")

            for item in trading_table:
                if item["report_type"] == 0:
                    report_file.write(
                        f"{item['date_time']}, {item['opening_price']}, {item['high_price']}, {item['low_price']}, {item['closing_price']}, {item['acc_price']}, {item['acc_volume']}\n"
                    )
                elif item["report_type"] == 1:
                    report_file.write(
                        f"{item['date_time']}, [->] {item['id']}, {item['type']}, {item['price']}, {item['amount']}\n"
                    )
                elif item["report_type"] == 2:
                    report_file.write(
                        f"{item['date_time']}, [<-] {item['request']['id']}, {item['type']}, {item['price']}, {item['amount']}, {item['msg']}\n"
                    )
                elif item["report_type"] == 3:
                    report_file.write(
                        f"{item['date_time']}, [#] {item['balance']}, {item['cumulative_return']}, {item['price_change_ratio']}, {item['asset']}\n"
                    )

            report_file.write("### SUMMARY =======================================\n")
            report_file.write(f"Property                 {summary[0]:10} -> {summary[1]:10}\n")
            report_file.write(
                f"Gap                                    {summary[1] - summary[0]:10}\n"
            )
            report_file.write(f"Cumulative return                    {summary[2]:10} %\n")
            report_file.write(f"Price_change_ratio {summary[3]}\n")

    def __draw_graph(self, filename, is_fullpath=False):
        total = pd.DataFrame(self.__create_plot_data())
        total = total.rename(
            columns={
                "date_time": "Date",
                "opening_price": "Open",
                "high_price": "High",
                "low_price": "Low",
                "closing_price": "Close",
                "acc_volume": "Volume",
            }
        )
        total = total.set_index("Date")
        total.index = pd.to_datetime(total.index)
        apds = []

        if "buy" in total.columns:
            apds.append(mpf.make_addplot(total["buy"], type="scatter", markersize=100, marker="^"))
        if "sell" in total.columns:
            apds.append(mpf.make_addplot(total["sell"], type="scatter", markersize=100, marker="v"))
        if "avr_price" in total.columns:
            apds.append(mpf.make_addplot(total["avr_price"]))
        if "return" in total.columns:
            apds.append(mpf.make_addplot((total["return"]), panel=1, color="g", secondary_y=True))

        destination = self.OUTPUT_FOLDER + filename + ".jpg"
        if is_fullpath:
            destination = filename

        mpf.plot(
            total,
            type="candle",
            volume=True,
            addplot=apds,
            mav=self.SMA,
            style="starsandstripes",
            savefig=dict(fname=destination, dpi=100, pad_inches=0.25),
        )
        return destination

    def __get_start_property_value(self):
        return round(self.__get_property_total_value(0))

    def __get_last_property_value(self):
        return round(self.__get_property_total_value(-1))

    def __get_property_total_value(self, index):
        total = float(self.asset_info_list[index]["balance"])
        quote = self.asset_info_list[index]["quote"]
        for name, item in self.asset_info_list[index]["asset"].items():
            total += float(item[1]) * float(quote[name])
        return total

    def __create_plot_data(self):
        result_pos = 0
        score_pos = 0
        last_avr_price = None
        last_acc_return = 0
        plot_data = []

        # 그래프를 그리기 위해 매매, 수익률 정보를 트레이딩 정보와 합쳐서 하나의 테이블로 생성
        self.logger.debug("report plot data ===================================")
        for info in self.info_list:
            new = info.copy()
            info_time = datetime.strptime(info["date_time"], self.ISO_DATEFORMAT)

            # 매매 정보를 생성해서 추가. 없는 경우 추가 안함. 기간내 매매별 하나씩만 추가됨
            while result_pos < len(self.result_list):
                result = self.result_list[result_pos]
                result_time = datetime.strptime(result["date_time"], self.ISO_DATEFORMAT)
                if info_time < result_time:
                    break

                if result["type"] == "buy":
                    new["buy"] = result["price"]
                elif result["type"] == "sell":
                    new["sell"] = result["price"]
                result_pos += 1

            # 수익률 정보를 추가. 정보가 없는 경우 최근 정보로 채움
            while score_pos < len(self.score_list):
                score = self.score_list[score_pos]
                score_time = datetime.strptime(score["date_time"], self.ISO_DATEFORMAT)

                # keep last one only
                if info_time >= score_time:
                    new["return"] = last_acc_return = score["cumulative_return"]
                    last_avr_price = None

                    if (
                        len(score["asset"]) > 0
                        and score["asset"][0][1] > 0  # 평균 가격
                        and score["asset"][0][3] > 0  # 현재 수량
                    ):
                        new["avr_price"] = last_avr_price = score["asset"][0][1]
                    score_pos += 1
                else:
                    new["return"] = last_acc_return
                    if last_avr_price is not None:
                        new["avr_price"] = last_avr_price
                    break
            plot_data.append(new)
        return plot_data
    

if __name__ == "__main__":
    
    '''
    1. 콜백 함수를 입력받아 초기화하고 (initialize)

    2. 시작시점 거래 정보를 기록 (make_start_point)
    
    3. 거래 정보를 저장하고 (put_trading_info)
    
    4. 거래 요청 정보를 저장 (put_request)
    
    5. 거래 결과 정보를 저장 (put_result)
    
    6. 자산 정보를 업데이트 (update_asset_info)
    
    7. 간단한 수익률 보고서 출력 (get_return_report)
    
    '''
    
    
    # 초기화 - 초기 자산 정보와 자산 정보 업데이트 함수 설정, 기준점 생성 
    analyzer = Analyzer()
    
    # asset_info 변수로 함수의 반환값을 설정하기 위한 테스트용 콜백 함수 
    def test_update_info_func(): # testing을 위해 hard cording 한 정보를 넣어줌
        asset_info = {
            "balance": 50000,
            "asset": {},
            "quote": {"KRW-BTC": 26145000.0},
            "date_time": "2020-12-21T01:13:00",
        }
        return asset_info
    
    # 1. 콜백 함수를 입력받아 초기화 (initialize)
    analyzer.initialize(test_update_info_func) # callback 함수를 입력받아 초기화함
    
    # 2. 시작시점 거래 정보를 기록 
    analyzer.make_start_point() 
    
    # put_trading_info가 호출된 후에 info_list가 어떻게 업데이트 되었는지 확인 
    info = {"market": "KRW-BTC",                # <= 원래는 data provider 에서 전달 받음
            "date_time": "2020-12-21T01:13:00",
            "opening_price": 26155000.0,
            "high_price": 26158000.0,
            "low_price": 26132000.0,
            "closing_price": 26145000.0,
            "acc_price": 116448937.40051,
            "acc_volume": 4.45311465,}
    # 3. 거래 정보를 저장 (put_trading_info)
    analyzer.put_trading_info(info) # 거래 정보를 저장함 
    analyzer.info_list # 거래 데이터 목록 
    
    # put_requests가 호출된 후에 request_list가 어떻게 업데이트 되었는지 확인 
    requests = [{"id": "1621767063.373",
                "type": "buy",
                "price": 26145000.0,
                "amount": 0.0004,
                "date_time": "2020-12-21T01:13:00",}]
    
    # 4. 거래 요청 정보를 저장 (put_request)
    analyzer.put_requests(requests)
    analyzer.request_list
    
    # put_result가 호출된 후에 result_list가 어떻게 업데이트 되었는지 확인
    result = {"request": {"id": "1621767063.373",
                        "type": "buy",
                        "price": 26145000.0,
                        "amount": 0.0004,
                        "date_time": "2020-12-21T01:13:00",
                        },
                        "type": "buy",
                        "price": 26145000.0,
                        "amount": 0.0004,
                        "msg": "success",
                        "balance": 39537,
                        "state": "done",
                        "date_time": "2020-12-21T01:13:00",}
    
    # 5. 거래 결과 정보를 저장 (put_result)
    analyzer.put_result(result)
    analyzer.result_list

    
    # test_update_info_func을 통해 전달된 자산 정보에 따라 asset_info_list가 어떻게 업데이트 되는지 확인
    def test_update_info_func():
        asset_info = {"balance": 39537,
                      "asset": {"KRW-BTC": (26145000.0, 0.0004)},
                      "quote": {"KRW-BTC": 26132000.0},
                      "date_time": "2020-12-21T01:14:00",}
        return asset_info

    analyzer.get_asset_info_func = test_update_info_func
    # 6. 자산 정보를 업데이트 (update_asset_info)
    analyzer.update_asset_info()
    analyzer.asset_info_list
    
    # 7. 간단한 수익률 보고서 출력 (get_return_report)
    # 보고서 결과 확인 (시작자산, 최종자산, 기준시점부터 누적 수익률, 기준시점부터 보유 종목별 가격 변동률 딕셔너리)
    analyzer.get_return_report()
    
    
    
    
    
    
    
    
    