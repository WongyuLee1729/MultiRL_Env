"""
시뮬레이션에 사용되는 모듈들을 연동하여 시뮬레이션을 운영

핵심적인 내용만 정리해보면.. 
1. Data Provider의 get_info를 통해서 거래 데이터를 가져와 Strategy의 update_trading_info와 Analyzer의 put_trading_info를 호출 

2. Strategy의 get_request를 통해서 거래 요청정보를 가져와 Trader의 send_request와 Analyzer의 put_requests으로 요청정보를 전달

3. 마지막으로, 콜백을 통해 Trader로부터 전달받은 거래 결과 데이터를 Strategy의 update_result와 Analyzer의 put_result를 통해 업데이트
 
이것이 자동거래 1회차 동안의 기본적인 동작이며, 이 동작을 정해진 주기마다 반복하는 것이 자동거래의 핵심 동작임

"""

from log_manager import LogManager
from engine_operator import Operator
import pdb

class SimulationOperator(Operator):
    """
    각 모듈을 연동해 시뮬레이션을 진행하는 클래스
    """

    def __init__(self):
        super().__init__()
        self.logger = LogManager.get_logger(__class__.__name__)
        self.turn = 0
        self.budget = 0

    def _execute_trading(self, task):
        pdb.set_trace()                 # Multi-threading debugger
        """자동 거래를 실행 후 타이머를 실행한다

        simulation_terminated 상태는 시뮬레이션에만 존재하는 상태로서 시뮬레이션이 끝났으나
        Operator는 중지되지 않은 상태. Operator의 시작과 중지는 외부부터 실행되어야 한다.
        
        1. Data Provider의 get_info를 통해서 거래 데이터를 가져와서 Strategy의 update_trading_info와 Analyzer의 put_trading_info를 호출 
        2. Strategy의 get_request를 통해 거래 요청 정보를 가져와 trader의 send_request와 Analyzer의 put_request로 요청정보 전달 
        3. 콜백을 통해 Trader로부터 전달받은 거래 결과 데이터를 Strategy의 update_result와 Analyzer의 put_result를 통해 업데이트 
        
        이것이 자동거래 1회차 동안의 기본적인 동작이며, 이 동작을 정해진 주기마다 반복하는 것이 자동거래의 핵심 동작임 
        
        """
        del task # engine_operator에서 생성한 task= {'runnable':method}는 시뮬레이션에서 사용하지 않으므로 제거함
        self.logger.info(f"############# Simulation trading is started : {self.turn + 1}")
        self.is_timer_running = False # 타이머가 동작하고 있는지 구분
        try: 
            ### 1. 
            trading_info = self.data_provider.get_info() # data_provider에서 api를 통해 가져온 데이터 -> 100개를 한번에 가져오면 그 중에 하나씩 가져오는건지..?
            self.strategy.update_trading_info(trading_info) # 외부에서 새로운 거래 정보를 Strategy에 전달하기 위한 인터페이스-> self.data에 데이터를 append 함 
            self.analyzer.put_trading_info(trading_info)    #거래 요청 및 결과에 대한 개괄적인 정보를 저장한다 0: 거래 데이터, 1: 매매 요청, 2: 매매 결과, 3: 수익률 정보
            # info = {'market': 'KRW-BTC', 'date_time': '2023-04-30T07:00:00', 
                    # 'opening_price': 39000000.0, 'high_price': 39004000.0, 'low_price': 38972000.0, 'closing_price': 38972000.0,
                    # 'acc_price': 46044759.07734, 'acc_volume': 1.18084374}
            def send_request_callback(result): # result = self.market.handle_request(request_list[0]) , self.trader.send_request 함수에서 callback 변수 함수로 send_request_callback 함수를 받음
                
                ''' 
                trader는 strategy로부터 거래 요청 정보를 받아 virtual_market에 보내 체결 내역을 받음. 여기서 이 정보를 받아오기 위해 콜백 함수를 사용함
                trader에서 사용되는 콜백 함수로 거래 결과는 비주기적으로 업데이트되기 때문에 trader에게 콜백과 함께 거래를 요청하며, 
                콜백 함수를 통해서 결과를 업데이트해 주는 방식 -> trader에서는 거래만 할 수 있게 거래와 요청 기능을 분리하기 위해 콜백 함수를 사용함 
                시뮬레이션의 경우 가상 거래소를 통해서 결과가 바로 전달되므로 Trader에서는 결과를 바로 콜백으로 호출해줌 
                실제 거래소와 연결되는 trader의 경우 콜백 정보와 요청 정보를 trader가 관리하고 있다가 거래소에서 체결 결과를 조회해서 콜백을 호출해 주어야 함 
                즉, trader는 요청된 거래를 virtual_market에 보내 체결 시키는 기능을 하며, 해 당 거래 정보를 다시 보내 주는 역할은 콜백함수를 이용해서 함 
                '''
                self.logger.debug("send_request_callback is called")
                if result == "error!":
                    self.logger.error("request fail")
                    return

                if result["msg"] == "game-over":
                    trading_info = self.data_provider.get_info()
                    self.analyzer.put_trading_info(trading_info)
                    self.last_report = self.analyzer.create_report(tag=self.tag)
                    self.state = "simulation_terminated"
                    return
                print('request successful !!')
                ### 3.
                self.strategy.update_result(result) # 요청한 거래의 결과를 업데이트
                self.analyzer.put_result(result)
            
            target_request = self.strategy.get_request() # 전략에 따라 거래 요청 정보를 생성 "id": 요청 정보(“1607862457.560075”), 
            ### 2. 
            if target_request is not None: # “type": 거래(sell/buy/cancel), “price": 거래 가격, “amount": 거래 수량”, date_time": 요청 데이터 생성 시간 (시뮬레이션 모드에서는 데이터 시간)
                self.trader.send_request(target_request, send_request_callback) # send_request(self, request_list, callback) <- 여기서 callback 함수로 send_request_callback을 받았음
                self.analyzer.put_requests(target_request)
        except AttributeError:
            self.logger.error("excuting fail")

        self.turn += 1
        # print(self.turn)
        self.logger.debug("############# Simulation trading is completed")
        self._start_timer()
        return True

    def get_score(self, callback):
        """현재 수익률을 인자로 전달받은 콜백함수를 통해 전달한다
        시뮬레이션이 종료된 경우 마지막 수익률 전달한다

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
            self.logger.debug("already terminated return last report")
            callback(self.last_report["summary"])
            return

        def get_score_callback(task):
            try:
                task["callback"](self.analyzer.get_return_report())
            except TypeError:
                self.logger.error("invalid callback")

        self.worker.post_task({"runnable": get_score_callback, "callback": callback})




if __name__ == "__main__":
    # 시뮬레이션에 사용될 객체 생성 및 초기화 
    import time
    from simulation_data_provider import SimulationDataProvider 
    # from simulation_operator import SimulationOperator
    from simulation_trader import SimulationTrader
    from strategy_bnh import StrategyBuyAndHold
    # import StrategySma0
    from analyzer import Analyzer
    import log_manager
    
    
    strategy = StrategyBuyAndHold()
    strategy.is_simulation = True 
    # 시뮬레이션 상태 설정 -> simulation에서는 시간 정보를 데이터 기준으로 사용하고, 
    # 다음 턴을 넘기기 위해서 가격과 수량을 0으로 하는 거래 요청 객체를 생성함
    
    end_date = "2023-04-30T07:30:00"
    end_str = end_date.replace(" ", "T")
    count = 19
    budget = 100000000
    interval = 1
    time_limit = 15 # 시간이 너무 오래 걸리는 문제를 검출하기 위해 통합 테스트의 시뮬레이션이 15초 안에 처리되는지 검증하기 위한 변수 

    '''
    SimulationOperator의 초기화 시점에 SimulationDataProvider와 SimulationTrader, Analyzer 객체까지 생성해서 전달해 주도록 했다.
    이렇게 외부에서 생성 후 전달하는 것이 Operator 내부에서 각 객체를 생성하는 것보다 좀 더 유연하게 모듈 조합을 구성할 수 있다. 
    '''
    data_provider = SimulationDataProvider()
    data_provider.initialize_simulation(end=end_str, count=count) # API를 통해 데이터를 가져옴 QUERY_STRING = {"market": "KRW-BTC"} 는 변수로 변경해줄 것!
    trader = SimulationTrader()
    trader.initialize_simulation(end=end_str, count=count, budget=budget)
    analyzer = Analyzer()
    analyzer.is_simulation = True

    # Simulation Operator 객체 생성 및 초기화 
    operator = SimulationOperator()
    operator.initialize(
        data_provider,
        strategy,
        trader,
        analyzer,
        budget=budget,
    )
    
    print(operator.state)
    
    
    '''
    초기화한 후에는 interval을 설정하고 자동거래를 시작한다. 테스트 케이스에서는 while 반복문을 돌리면서 operator의 상태 변수를 체크한다.
    simulation operator의 경우 시뮬레이션이 끝나면 상태 변수가 "simulation_terminated"로 변경되는 것을 이용한 것이다. 
    자동거래가 끝나면 get_trading_results 메서드를 이용해서 거래 결과 목록을 가져온 다음 예상 결과와 동일한지 검증한다. 
    '''
    
    # trading_snapshot = simulation_data.get_data("bnh_snapshot")
    # operator.set_interval(interval)
    # operator.start()
    # start_time = time.time()
    # while operator.state == "running":
    #     time.sleep(0.5)
    #     if time.time() - start_time > time_limit:
    #         self.assertTrue(False, "Time out")
    #         break
    # trading_results = operator.get_trading_results()
    # self.check_equal_results_list(trading_results, trading_snapshot)
    

    # Simulation 간격 설정 및 자동 거래 시작 
    operator.set_interval(interval)
    operator.start()
    
    # Simulation Operator 상태 확인 
    print(operator.state)
    
    # Simulation 거래 결과 목록 출력 
    operator.get_trading_results()
    
    # 시뮬레이션 수익률 보고서 출력 
    def callback(return_report):
        print(return_report)

    operator.get_score(callback)






