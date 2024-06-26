
from abc import ABCMeta, abstractmethod


class DataProvider(metaclass=ABCMeta):
    """
    Collect data from the data source and offers in in right format
    """

    @abstractmethod
    def get_info(self):
        """
        Deliver the current trading info

        Returns: 
        {
            "market": 
            "date_time": 
            "opening_price": 
            "high_price": 
            "low_price": 
            "closing_price": 
            "acc_price": 
            "acc_volume": 
        }
        """



