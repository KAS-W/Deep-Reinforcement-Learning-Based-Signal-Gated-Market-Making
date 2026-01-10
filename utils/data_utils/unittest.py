from tusharesql import DailyQueryTool, AShareQueryTool

tool = DailyQueryTool()
ashare = AShareQueryTool()

def test_calendar(start_date, end_date):
    cal_df = tool.us_trade_cal(start_date=start_date, end_date=end_date)

    # if dataframe is empty
    assert not cal_df.empty, f"Error: DataFrame is empty for dates {start_date} to {end_date}"

    # if dataframe has all columns
    expected_columns = ['cal_date', 'is_open', 'pretrade_date']
    for col in expected_columns:
        assert col in cal_df.columns, f"Error: Missing expected column '{col}'"
        
    print(f"Test Passed: Calendar data is valid with {len(cal_df)} rows.")
    return True

def test_daily(start_date, end_date):
    daily_df = tool.us_daily(start_date=start_date, end_date=end_date, trade_date='20250115')

    # if dataframe is empty
    assert not daily_df.empty, f"Error: DataFrame is empty for dates {start_date} to {end_date}"

    # if dataframe has all columns
    expected_columns = ['close', 'open', 'high', 'low', 'pre_close', 'pct_change', 'vol', 'amount', 'vwap']
    for col in expected_columns:
        assert col in daily_df.columns, f"Error: Missing expected column '{col}'"

    print(f"Test Passed: Daily data is valid with {len(daily_df)} rows.")
    # print(daily_df.columns)
    # print(daily_df.head(10))
    return True

def test_adjfactor(start_date, end_date):
    adjfactor_df = tool.us_adjfactor(start_date=start_date, end_date=end_date, trade_date='')

    # if dataframe is empty
    assert not adjfactor_df.empty, f"Error: DataFrame is empty for dates {start_date} to {end_date}"

    # if dataframe has all columns
    expected_columns = ['trade_date', 'exchange', 'cum_adjfactor', 'close_price']
    for col in expected_columns:
        assert col in adjfactor_df.columns, f"Error: Missing expected column '{col}'"
        
    print(f"Test Passed: Adjfactor data is valid with {len(adjfactor_df)} rows.")
    return True

def test_adj_daily(start_date, end_date):
    adj_daily = tool.us_daily_adj(start_date=start_date, end_date=end_date, trade_date='')

    # if dataframe is empty
    assert not adj_daily.empty, f"Error: DataFrame is empty for dates {start_date} to {end_date}"

    # if dataframe has all columns
    expected_columns = ['close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_change', 'vol', 'amount', 'vwap', 'turnover_ratio', 'adj_factor', 'free_share', 'free_mv', 'total_mv', 'exchange', 'total_share']
    for col in expected_columns:
        assert col in adj_daily.columns, f"Error: Missing expected column '{col}'"

    print(f"Test Passed: Daily adj data is valid with {len(adj_daily)} rows.")
    return True


# tushare daily() api has been verifed previously
# so here I only test the minute data api
# Tushare does not support cross-sectional snapshot for the minute level data
# for loops on trade_date and ts_code are required
def test_a_share_minute(start_date, end_date):
    ashare_minute = ashare.stk_mins(ts_code='510300.SH', start_date=start_date, end_date=end_date, freq='5min')
    
    # if dataframe is empty
    assert not ashare_minute.empty, f"Error: DataFrame is empty for dates {start_date} to {end_date}"

    expected_columns = ['close', 'open', 'high', 'low', 'vol', 'amount']
    for col in expected_columns:
        assert col in ashare_minute.columns, f"Error: Missing expected column '{col}'"

    print(f"Test Passed: Daily adj data is valid with {len(ashare_minute)} rows.")
    print(ashare_minute.columns)
    print(ashare_minute.head(10))
    return True


if __name__ == '__main__':
    start_date = '20250101'
    end_date = '20250201'

    # us data test
    # test_calendar(start_date, end_date)
    # test_daily(start_date, end_date)
    # test_adjfactor(start_date, end_date)
    # test_adj_daily(start_date, end_date)

    # a-share minute data test
    test_a_share_minute(start_date, end_date)