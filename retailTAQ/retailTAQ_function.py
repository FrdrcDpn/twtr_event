import wrds

from retailTAQ import taq_daily

def initTAQ():
    # initialise our wrds sql server
    print('Initialising wrds ... (1/8)')
    taq_object = taq_daily.TaqDaily(method='PostgreSQL', db=wrds.Connection(wrds_username='fredericdupon'),
                                         track_retail=True)
    return taq_object


def get_nbbo_df(taq_object, time, symbol):
    # get our national best bid order table for our given time
    print('Retrieving national best bid ask table ... (2/8)')
    nbbo_table = taq_object.get_nbbo_table(time, symbols=symbol, common_only=True)
    return nbbo_table


def get_quote_df(taq_object, time, symbol):
    print('Retrieving quote table ... (3/8)')
    quote_table = taq_object.get_quote_table(time, symbols=symbol, common_only=True)
    return quote_table


def get_trades_df(taq_object, time, symbol):
    print('Retrieving trades table ... (4/8)')
    trade_table = taq_object.get_trade_table(time, symbols=symbol, common_only=True)
    return trade_table


def get_offical_complete_nbbo(taq_object, time, symbol, nbbo_df, quote_df):
    print('Creating official complete national best bid ask table ... (5/8)')
    official_complete_nbbo_df = taq_object.get_official_complete_nbbo(date=time, symbols=symbol,
                                                                           nbbo_df=nbbo_df,
                                                                           quote_df=quote_df)
    return official_complete_nbbo_df


def merge_trades_nbbo(taq_object, trade_df, official_complete_nbbo_df):
    print('Merging trades table with national best bid ask table ... (6/8)')
    merged_trades_nbbo = taq_object.merge_trades_nbbo(trade_df, official_complete_nbbo_df, track_retail=True)
    return merged_trades_nbbo

# export to excel
# print('Exporting merged table to excel ... (7/8)')
# merged_trades_nbbo.to_excel("output.xlsx")

# export to csv
# print('Exporting merged table to csv ... (8/8)')
# merged_trades_nbbo.to_csv('output.csv')

# print('DONE!!!')
# some other metrics to be calculated
# spreads = taq_object.compute_spreads(time, official_complete_nbbo_table, start_time_spreads=None, end_time_spreads=None)
# effective_spreads = taq_object.compute_effective_spreads(merged_trades_nbbo)
# rs_and_pi = taq_object.compute_rs_and_pi(merged_trades_nbbo, official_complete_nbbo_table, delay, suffix,track_retail=None)
# weighted_average = taq_object.compute_averages_ave_sw_dw(self, df, measures, simple=True,dollar_weighted=True,
# share_weighted=True)

# print(quote_table)
# print(taq_object.get_nbbo_table_postgresql( time, symbols='TSLA'))
