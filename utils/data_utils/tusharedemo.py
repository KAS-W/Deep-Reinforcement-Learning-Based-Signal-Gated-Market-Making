from typing import List
import tushare as ts
import pandas as pd
import traceback
from pandas import DataFrame

class DateError(ValueError):
    """Error in trading date"""
    pass

class DailyQueryTool:
    """
    Wrapping up Tushare SDK and redirect to private server
    """
    
    def __init__(self):
        token = ''
        self.pro = ts.pro_api(token='')
        # Name Mangling princial for private server
        self.pro._DataApi__token = token
        self.pro._DataApi__http_url = ''

    def __general_api_query(self, api_name:str, **kwargs) -> DataFrame:
        """
        MySql API for query
        """
        try:
            df = self.pro.query(api_name, **kwargs)
            if df is not None:
                return df.drop_duplicates()
            return pd.DataFrame()
        except Exception:
            # print error msg and return SQL/param
            traceback.print_exc()
            print(f"API Error: [{api_name} with params: {kwargs}]")
            raise ValueError(f"Failed to fetch data from custom API: {api_name}")
        
    def __check_date_logic(self, start_date: str = None, end_date: str = None):
        """verify if the start date is later than the end date"""
        if start_date and end_date:
            # yyyymmdd
            s = int(str(start_date).replace('-', ''))
            e = int(str(end_date).replace('-', ''))
            if s > e:
                raise DateError(f"start_date={start_date} > end_date={end_date}")
            
    def us_basic(self, ts_code:str='', list_status:str=None) -> DataFrame:
        """
        Return basic info of US equity market
        """
        return self.__general_api_query('us_basic', ts_code=ts_code, list_status=list_status)
    
    def us_trade_cal(self, start_date:str='20250101', end_date:str='20250201', is_open:str='1') -> DataFrame:
        """
        Return US market trading calendar
        """
        self.__check_date_logic(start_date, end_date)
        return self.__general_api_query('us_tradecal', start_date=start_date, end_date=end_date, is_open=is_open)
    
    def us_daily(self, ts_code:str='', trade_date:str='20250101', start_date:str='20250101', end_date:str='20250201') -> DataFrame:
        """Return US market daily"""
        self.__check_date_logic(start_date, end_date)
        return self.__general_api_query('us_daily', ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date)
    
    def us_daily_adj(self, ts_code:str='', trade_date:str='20250101', start_date:str='20250101', end_date:str='20250201', exchange:str='', offset:int=0, limit:int=5000) -> DataFrame:
        """Return adjusted US market daily"""
        self.__check_date_logic(start_date, end_date)
        return self.__general_api_query('us_daily_adj', ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, exchange=exchange, offset=offset, limit=limit)
    
    def us_adjfactor(self, ts_code:str='', trade_date:str='20250101', start_date:str='20250101', end_date:str='20250201') -> DataFrame:
        """Return adjust factor"""
        self.__check_date_logic(start_date, end_date)
        return self.__general_api_query('us_adjfactor', ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date)
    
class AShareQueryTool:
    """Python SDK for A share Minute Level Data"""

    def __init__(self):
        token = ''
        self.pro = ts.pro_api(token='')
        # 使用 Name Mangling 规则访问私有成员
        self.pro._DataApi__token = token
        self.pro._DataApi__http_url = ''

    def __general_api_query(self, api_name:str, **kwargs) -> DataFrame:
        """
        MySql API for A-Share query
        """
        try:
            df = self.pro.query(api_name, **kwargs)
            if df is not None:
                return df.drop_duplicates()
            return pd.DataFrame()
        except Exception:
            traceback.print_exc()
            print(f'API Interface Error: [{api_name} with params: {kwargs}]')
            raise ValueError(f"Failed to fetch data from custom API: {api_name}")
        
    def __check_date_logic(self, start_date: str = None, end_date: str = None):
        """verify if the start date is later than the end date"""
        if start_date and end_date:
            # yyyymmdd
            s = int(str(start_date).replace('-', ''))
            e = int(str(end_date).replace('-', ''))
            if s > e:
                raise DateError(f"start_date={start_date} > end_date={end_date}")

    def daily(self, ts_code:str=None, trade_date:str=None, start_date:str=None, end_date:str=None) -> DataFrame:
        """A Share Daily"""
        self.__check_date_logic(start_date, end_date)
        return self.__general_api_query('daily', ts_code=ts_code, trade_date=trade_date,
                                         start_date=start_date, end_date=end_date)
    
    def daily_basic(self, ts_code:str=None, trade_date:str=None, start_date:str=None, end_date:str=None) -> DataFrame:
        """Daily Indicator"""
        self.__check_date_logic(start_date, end_date)
        return self.__general_api_query('moneyflow', ts_code=ts_code, trade_date=trade_date,
                                         start_date=start_date, end_date=end_date)
    
    def skt_limit(self, ts_code:str=None, trade_date:str=None, start_date:str=None, end_date:str=None) -> DataFrame:
        """Daily Limit"""
        self.__check_date_logic(start_date, end_date)
        return self.__general_api_query('skt_limit', ts_code=ts_code, trade_date=trade_date,
                                         start_date=start_date, end_date=end_date)
    
    def stk_mins(self, ts_code:str=None, freq:str='1min', start_date:str=None, end_date:str=None) -> DataFrame:
        """
        Stock Minute 

        :param freq: 1min, 5min, 15min, 30min, 60min
        :param start_date: '2025-01-01 09:30:00'
        :param end_date: '2025-01-01 15:00:00'
        """
        return self.__general_api_query('stk_mins', ts_code=ts_code, freq=freq, 
                                       start_date=start_date, end_date=end_date)
    
    def index_mins(self, ts_code:str=None, freq:str='1min', start_date:str=None, end_date:str=None) -> DataFrame:
        """Index Minute"""
        return self.__general_api_query('index_mins', ts_code=ts_code, freq=freq, 
                                       start_date=start_date, end_date=end_date)
    
    def stock_basic(self, ts_code: str = '', market: str = None, list_status: str = None, exchange: str = None, is_hs: str = None, fields: str = 'ts_code, symbol, name, area, industry, list_date') -> DataFrame:
        """Stock List Info"""
        return self.__general_api_query('stock_basic', ts_code=ts_code, market=market, 
                                       list_status=list_status, exchange=exchange, is_hs=is_hs, fields=fields)

    def trade_cal(self, exchange: str = '', start_date: str = None, end_date: str = None, is_open: str = None) -> DataFrame:
        """Trdaing Calendar"""
        return self.__general_api_query('trade_cal', exchange=exchange, start_date=start_date, 
                                       end_date=end_date, is_open=is_open)

    def namechange(self, ts_code: str = None, start_date: str = None, end_date: str = None) -> DataFrame:
        """Stock Name"""
        return self.__general_api_query('namechange', ts_code=ts_code, start_date=start_date, end_date=end_date)

    def suspend_d(self, ts_code: str = None, trade_date: str = None, suspend_type: str = None) -> DataFrame:
        """Stock Suspend"""
        return self.__general_api_query('suspend_d', ts_code=ts_code, trade_date=trade_date, suspend_type=suspend_type)