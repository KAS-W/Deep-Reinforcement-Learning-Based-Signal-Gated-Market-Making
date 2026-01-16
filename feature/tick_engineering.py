import numpy as np

def v_impluse(tick_df):
    tick_df.loc[tick_df['trade_time'] < 93000000, 'trade_time'] = 93000000
    
    tick_df['opp_vol'] = np.where(tick_df['side'] == 1,
                            tick_df['SaleOrderVolume'],
                            tick_df['BuyOrderVolume'])

    tick_df['f_exhaustion'] = tick_df['Volume'] / (tick_df['opp_vol'] + 1e-9)

    tick_df['opp_price'] = np.where(tick_df['side'] == 1,
                            tick_df['SaleOrderPrice'],
                            tick_df['BuyOrderPrice'])

    tick_df['f_p_aggression'] = (tick_df['Price'] - tick_df['opp_price']) * tick_df['side']

    unique_sale_ids = tick_df.groupby('trade_time')['SaleOrderID'].transform('nunique')
    unique_buy_ids = tick_df.groupby('trade_time')['BuyOrderID'].transform('nunique')

    tick_df['is_sweep'] = np.where(tick_df['side'] == 1,
                                   unique_sale_ids > 1,
                                   unique_buy_ids > 1).astype(int)

    tick_agg = tick_df.groupby('trade_time').agg({
            'f_exhaustion': 'mean',
            'f_p_aggression': 'sum',
            'is_sweep': 'mean',  
            'Volume': 'sum',
            'side': 'mean',
        }).reset_index()

    tick_agg['f_tick_flow'] = tick_agg['side'] * (1 + tick_agg['is_sweep']) * tick_agg['f_exhaustion'] * np.log1p(tick_agg['Volume'])

    return tick_agg