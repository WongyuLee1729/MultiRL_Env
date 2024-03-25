"""시스템의 운영을 담당

이 모듈은 각 모듈을 컨트롤하여 전체 시스템을 운영한다.
"""

import time
import threading
from datetime import datetime
from log_manager import LogManager
from worker import Worker


class Operator:
    """
    전체 시스템의 운영을 담당하는 클래스

    Attributes:
        data_provider: 사용될 DataProvider 인스턴스
        strategy: 사용될 Strategy 인스턴스
        trader: 사용될 Trader 인스턴스
        analyzer: 거래 분석용 Analyzer 인스턴스
        interval: 매매 프로세스가 수행되는 간격 # default 10 second
    """

    ISO_DATEFORMAT = "%Y-%m-%dT%H:%M:%S"
    OUTPUT_FOLDER = "output/"

    def __init__(self):
        self.logger = LogManager.get_logger(__class__.__name__)
        self.data_provider = None
        self.strategy = None
        self.trader = None
        self.interval = 10  # default 10 second
        self.is_timer_running = False # 정해놓은 거래 빈도시간 이후에 거래될 수 있도록 timer를 설정, timer_running=True라면 거래하지 않음
        self.timer = None
        self.analyzer = None
        self.worker = Worker("Operator-Worker") # operator worker를 초기화해 줌 
        self.state = None # 초기화가 정상적으로 진행되면 상태변수 state를 "ready"로 변경해줌 
        self.is_trading_activated = False
        self.tag = datetime.now().strftime("%Y%m%d-%H%M%S") # Analyzer에서 생성되는 기록에 사용되는 값, 기록 데이터에서 operator를 식별하기 위해 사용 
        self.timer_expired_time = datetime.now()
        self.last_report = None

    def initialize(self, data_provider, strategy, trader, analyzer, budget=500):
        """
        운영에 필요한 모듈과 정보를 설정 및 각 모듈 초기화 수행

        data_provider: 운영에 사용될 DataProvider 객체
        strategy: 운영에 사용될 Strategy 객체
        trader: 운영에 사용될 Trader 객체
        analyzer: 운영에 사용될 Analyzer 객체
        """
        if self.state is not None:
            return

        self.data_provider = data_provider
        self.strategy = strategy
        self.trader = trader
        self.analyzer = analyzer
        self.state = "ready"
        self.strategy.initialize(budget)
        self.analyzer.initialize(trader.get_account_info)

    def set_interval(self, interval):
        """자동 거래 시간 간격을 설정한다.

        interval : 매매 프로세스가 수행되는 간격
        """
        self.interval = interval

    def start(self):
        """자동 거래를 시작한다

        자동 거래는 설정된 시간 간격에 맞춰서 Worker를 사용해서 별도의 스레드에서 처리된다.
        
        1. start 메서드가 호출되면 worker가 _executing_trading 메서드를 대신 실행해줌
        2. _executing_trading 에서 거래를 실행하고 _start_timer를 호출하고 여기서 threading의 Timer를 이용해 타이머 시작 
            즉, _start_timer는 타이머를 재시작하는 함수임 
            
        
        """
        if self.state != "ready":
            return False

        if self.is_timer_running:
            return False

        self.logger.info("===== Start operating =====")
        self.state = "running"
        self.analyzer.make_start_point() # 시작점의 수익률을 기록함 
        self.worker.start() # start 메서드를 호출 
        self.worker.post_task({"runnable": self._execute_trading}) # 여기서 처음으로 worker의 task_queue에 task를 넣고, _execute_trading을 실행함
        # 여기서 _execute_trading은 simulation_operator에서 overriding한 함수임 -> sim operation으로 넘어가서 현재 task를 지우고 
        # 해 당 _execute_trading를 실행 함
        self.tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            self.tag += "-" + self.trader.name + "-" + self.strategy.name # + "-" + self.trader.MARKET
        except AttributeError:
            self.logger.warning("can't get additional info form strategy and trader")
        return True

    def _start_timer(self):
        """
        설정된 간격의 시간이 지난 후 Worker가 자동 거래를 수행하도록 타이머 설정
        
        on_timer_expired 라는 내장 함수를 이용해 타이머를 시작함 
        타이머가 만료되었을 때 다시 worker가 _executing_trading 메서드를 실행하도록 하는게 전부임
        
        """
        self.logger.debug(
            f"start timer {self.is_timer_running} : {self.state} : {threading.get_ident()}"
        )
        if self.is_timer_running or self.state != "running":
            return

        def on_timer_expired():
            self.timer_expired_time = datetime.now()
            self.worker.post_task({"runnable": self._execute_trading})

        adjusted_interval = self.interval # 타이머 만료 시점부터 execute_trading이 수행되고 다시 _start_timer 메서드 실행 시점 시간을 interval 변수에서 뺀 값
        # 예를들어 1분마다 자동거래가 실행되는데, 타이머가 만료되어 자동거래 처리에 5초가 걸렸다면 다음 타이머는 55초로 설정해야 시간이 밀리지 않고 의도한 1분 자동거래가 실행 됨
        if self.interval > 1:
            time_delta = datetime.now() - self.timer_expired_time
            adjusted_interval = self.interval - round(time_delta.total_seconds(), 1)

        self.timer = threading.Timer(adjusted_interval, on_timer_expired)
        # threading.Timer는 파이썬 기본 기능으로 초단위 시간과 호출 함수를 인자로 호출할 경우 그 시간 이후 전달 받은 함수를 호출해줌
        self.timer.start()

        self.is_timer_running = True
        return

    def _execute_trading(self, task):
        """
        자동 거래를 실행 후 타이머를 실행한다
        
        1. Data Provider의 get_info를 통해서 거래 데이터를 가져와서 Strategy의 update_trading_info와 Analyzer의 put_trading_info를 호출 
        2. Strategy의 get_request를 통해 거래 요청 정보를 가져와 trader의 send_request와 Analyzer의 put_request로 요청정보 전달 
        3. 콜백을 통해 Trader로부터 전달받은 거래 결과 데이터를 Strategy의 update_result와 Analyzer의 put_result를 통해 업데이트 
        
        이것이 자동거래 1회차 동안의 기본적인 동작이며, 이 동작을 정해진 주기마다 반복하는 것이 자동거래의 핵심 동작임 
        """
        del task
        self.logger.debug("trading is started #####################")
        self.is_timer_running = False # 타이머가 동작하고 있는지 확인 
        try:
            trading_info = self.data_provider.get_info()
            self.strategy.update_trading_info(trading_info)
            self.analyzer.put_trading_info(trading_info)
            self.logger.debug(f"trading_info {trading_info}")

            def send_request_callback(result): # Trader로부터 거래결과를 전달받을 때 사용될 콜백 함수 
                self.logger.debug("send_request_callback is called")
                if result == "error!":
                    self.logger.error("request fail")
                    return
                self.strategy.update_result(result)

                if "state" in result and result["state"] != "requested":
                    self.analyzer.put_result(result)

            target_request = self.strategy.get_request()
            self.logger.debug(f"target_request {target_request}")
            if target_request is not None:
                self.trader.send_request(target_request, send_request_callback)
                self.analyzer.put_requests(target_request)
        except (AttributeError, TypeError) as msg:
            self.logger.error(f"excuting fail {msg}")

        self.logger.debug("trading is completed #####################")
        self._start_timer()
        return True

    def stop(self):
        """거래를 중단한다"""
        if self.state != "running":
            return

        self.trader.cancel_all_requests()
        trading_info = self.data_provider.get_info()
        self.analyzer.put_trading_info(trading_info)
        self.last_report = self.analyzer.create_report(tag=self.tag)
        self.logger.info("===== Stop operating =====")
        try:
            self.timer.cancel()
        except AttributeError:
            self.logger.error("stop operation fail")
        self.is_timer_running = False
        self.state = "terminating"

        def on_terminated():
            self.state = "ready"

        self.worker.register_on_terminated(on_terminated)
        self.worker.stop()

    def get_score(self, callback):
        """현재 수익률을 인자로 전달받은 콜백함수를 통해 전달한다
        
        자동거래가 수행되는 동안 조회시점의 수익률을 조회해 볼 수 있는 기능으로 
        앞에서와 마찬가지로 Operator 자체 로직이 있는 것이 아니라 Analyzer에서 구현한 get_return_report 메서드를 
        실행하고 결괏값을 전달하는 것이 Operator의 역할임. 
        전달과정에서 직접 Analyzer의 get_return_report를 호출하는 것이 아닌 Worker를 통해 실행하여 별도의 스레드에서 동작하도록함
        -> 이를 위해 콜백 내부 함수를 만듦
        
        
        Returns:
            (
                start_budget: 시작 자산
                final_balance: 최종 자산
                cumulative_return : 기준 시점부터 누적 수익률
                price_change_ratio: 기준 시점부터 보유 종목별 가격 변동률 딕셔너리
                graph: 그래프 파일 패스
            )
        """

        if self.state != "running":
            self.logger.warning(f"invalid state : {self.state}")
            return

        def get_score_callback(task):
            graph_filename = f"{self.OUTPUT_FOLDER}g{round(time.time())}.jpg"
            try:
                task["callback"](self.analyzer.get_return_report(graph_filename))
            except TypeError:
                self.logger.error("invalid callback")

        self.worker.post_task({"runnable": get_score_callback, "callback": callback})

    def get_trading_results(self):
        """현재까지 거래 결과 기록을 반환한다"""
        return self.analyzer.get_trading_results()












