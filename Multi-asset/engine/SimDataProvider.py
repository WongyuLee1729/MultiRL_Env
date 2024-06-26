
from datetime import datetime, timedelta
from DataProvider import DataProvider
from log_manager import LogManager
from DataRepository import DataRepository


class SimulationDataProvider(DataProvider):
    """
    Collect data from exchange and offers it in sequence
    """

    AVAILABLE_CURRENCY = {"BTC": "KRW-BTC", "ETH": "KRW-ETH", "DOGE": "KRW-DOGE"} # change it!

    def __init__(self, currency="BTC"):
        if currency not in self.AVAILABLE_CURRENCY:
            raise UserWarning(f"not supported currency: {currency}")
        self.logger = LogManager.get_logger(__class__.__name__)
        self.repo = DataRepository("Engine.db")
        self.data = []
        self.index = 0

        self.market = self.AVAILABLE_CURRENCY[currency]

    def initialize_simulation(self, end=None, count=100):
        """
        Bring the data through DataRepository and initalize them
        """
        self.index = 0
        end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")
        start_dt = end_dt - timedelta(minutes=count)
        start = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        self.data = self.repo.get_data(start, end, market=self.market)
        # print(len(self.data))
        
    def get_info(self):
        """
        Deliver data in sequential order 

        Returns:
        {
            "market": trading market types BTC, ETH ..
            "date_time": base time 
            "opening_price": candle
            "high_price": candle
            "low_price": candle
            "closing_price": candle
            "acc_price": running of total price in a given time 
            "acc_volume": running of total volume in a given time 
        }
        """
        now = self.index
 
        if now >= len(self.data):
            return None
        # print('now:', now)
        self.index = now + 1
        self.logger.info(f'[DATA] @ {self.data[now]["date_time"]}')
        return self.data[now] #  self.data[now]


if __name__ == '__main__':
    
    end_date = "2023-12-20T07:30:00" 
    dr = SimulationDataProvider()    
    dr.initialize_simulation(end=end_date, count=10000)

    data1 = dr.get_info()
    for _ in range(10000):
        # 첫번째 데이터 
        print(data1)

    # end_date = "2020-04-30T07:30:00"
    # dp = SimulationDataProvider()  
    # dp.initialize_simulation(end=end_date, count=100)

    # data2 = dp.get_info()
    
    # # 두번째 데이터 
    # print(data2)


