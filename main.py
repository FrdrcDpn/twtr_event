# general imports
from time import mktime
import datetime
import tikzplotlib as mtikz
import matplotlib.ticker as mtick
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import YahooFinanceScraper.YahooFinance_functions as yh
import retailTAQ.retailTAQ_function as TAQ
import GeneralCode.Gernal_functions as gf
import TwitterScraper.twitterscraper_functions as twitter
from nltk.corpus import stopwords
import nltk.sentiment.vader as vd
from sklearn import preprocessing
from stargazer.stargazer import Stargazer, LineLocation
import matplotlib.pyplot as plt
import numpy as np
import wrds

# Set desired options (debug only)
pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 1000)  # or 1000
pd.set_option('display.max_colwidth', 190009)  # or 199
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore')

# data download/import preferences
download_elon_twitter_data = False
download_TAQ_data = False
download_Yahoo_data = False
download_twitter_sentiment_data = False
download_smp_500_data = False
download_fama_risk_factors = False

# process preferences
process_elon_twitter_data = False
process_twitter_sentiment_data = False
process_TAQ_data = False
process_Yahoo_data = False
process_smp500_data = False

# analysis
generate_analysis_dataset = False
granger_analysis = False
cumulative_return_analysis = False
fama_cumulative_return_analysis = False
simple_regression_analysis = False
multivariate_regression_analysis = False
fama_cross_sectional_regression_analysis = False
simple_multivariate_regression_analysis = True
simple_fama_cross_sectional_regression_analysis = True
# plotting preferences
plot_elon_twitter_behaviour = False
plot_TAQ_data = False
plot_cumulative_return = False
plot_cumulative_return_fama = False
plot_RH_data = False

# exporting preferences
export_tikz = False
export_discrete_events_latex = False
export_granger_analysis_latex = False
export_cumul_analysis_latex = False

fama_df_basic = pd.DataFrame()
fama_df_full_no_interact = pd.DataFrame()
fama_df_full = pd.DataFrame()
simple_fama_df_full = pd.DataFrame()
# date symbol company_name sentiment/result
events = [["01/04/18", "TSLA", "tesla", 0],
          ["14/05/18", "TSLA", "tesla", 0],
          ["07/08/18", "TSLA", "tesla", 1],
          ["01/05/20", "TSLA", "tesla", 0],
          ["26/01/21", "GME", "gamestop", 1],
          ["26/01/21", "ETSY", "etsy", 1],
          ["28/01/21", "CDR.WA", "cyberpunk", 1],
          ["13/06/21", "BTC-USD", "bitcoin", 1],
          ["14/10/21", "GOGO", "starlink", 1],
          ["01/11/21", "HTZ", "hertz", 1],
          ["06/11/21", "TSLA", "tesla", 0],
          ["14/04/22", "TSLA", "tesla", 0],
          ["25/04/22", "TWTR", "twitter", 1],
          ["13/05/22", "TWTR", "twitter", 0],
          ["18/05/22", "TSLA", "twitter", 0]]

# intialisation of some global variables or arrays or preferences
full_regressions = []
full_regressions_fama = []

vif_tables = pd.DataFrame()
maxtweets = 10000  # maximum number of tweets/day to be used for discrete event sentiment analysis
wrdsconn = wrds.Connection(wrds_username="fredericdupon")

# ********** SELECT INDIVIDUAL EVENT **********
# single event analysis not in our discrete event loop
event_id = 0
# retrieval from discrete event array
symbol = events[event_id][1]  # ticker symbol
date = events[event_id][0]  # date
keywords = events[event_id][2]  # keywords associated with event ; usually company name
savedate = date.replace("/", "_")  # replace forward slashes with underscore for saving
start_date = datetime.datetime.strptime(date, '%d/%m/%y') - datetime.timedelta(days=31)
discrete_date = datetime.datetime.strptime(date, '%d/%m/%y')
end_date = datetime.datetime.strptime(date, '%d/%m/%y') + datetime.timedelta(days=31)

# start of program
if __name__ == '__main__':

    # we loop over our discrete event array
    for event_id in range(0, 15):

        # discrete event information
        symbol = events[event_id][1]  # ticker symbol
        date = events[event_id][0]  # date
        keywords = events[event_id][2]  # keywords associated with event ; usually company name

        # print at which event we are
        print("DISCRETE EVENT:")
        print(date)

        # download data for date 1 month before and 1 month after the event
        savedate = date.replace("/", "_")  # replace forward slashes with underscore for saving
        start_date = datetime.datetime.strptime(date, '%d/%m/%y') - datetime.timedelta(days=31)
        discrete_date = datetime.datetime.strptime(date, '%d/%m/%y')
        end_date = datetime.datetime.strptime(date, '%d/%m/%y') + datetime.timedelta(days=31)

        # download TAQ data from wharton research data services
        if download_TAQ_data:
            print('Downloading TAQ data')

            # check if there is TAQ data available for the desired period
            if end_date < datetime.datetime(2019, 12, 31):
                # copy and store start date
                taq_start_date = start_date
                # check if we are retrieving data for a weekday when the markets are open otherwise add day
                while not taq_start_date.isoweekday() in range(1, 6):
                    taq_start_date += datetime.timedelta(days=1)

                # initialise wharton database
                test = TAQ.initTAQ()
                time = taq_start_date

                # retrieve data
                try:
                    nbbo_df = TAQ.get_nbbo_df(test, time, [symbol])
                    quote_df = TAQ.get_quote_df(test, time, [symbol])
                    trades_df = TAQ.get_trades_df(test, time, [symbol])
                    off_nnbo_df = TAQ.get_offical_complete_nbbo(test, time, [symbol], nbbo_df, quote_df)
                    old_df = TAQ.merge_trades_nbbo(test, trades_df, off_nnbo_df)

                # if an exception is thrown when the markets are not open on for example a holiday,
                # increase day with 1 and try again
                except:
                    taq_start_date += datetime.timedelta(days=1)
                    nbbo_df = TAQ.get_nbbo_df(test, time, [symbol])
                    quote_df = TAQ.get_quote_df(test, time, [symbol])
                    trades_df = TAQ.get_trades_df(test, time, [symbol])
                    off_nnbo_df = TAQ.get_offical_complete_nbbo(test, time, [symbol], nbbo_df, quote_df)
                    old_df = TAQ.merge_trades_nbbo(test, trades_df, off_nnbo_df)
                    pass

                # retrieve for every date in our defined date-period
                while taq_start_date < end_date:
                    taq_start_date += datetime.timedelta(days=1)

                    # check if we are retrieving data for a weekday when the markets are open
                    while not taq_start_date.isoweekday() in range(1, 6):
                        taq_start_date += datetime.timedelta(days=1)

                    time = taq_start_date
                    # retrieve data
                    try:
                        nbbo_df = TAQ.get_nbbo_df(test, time, [symbol])
                        quote_df = TAQ.get_quote_df(test, time, [symbol])
                        trades_df = TAQ.get_trades_df(test, time, [symbol])
                        off_nnbo_df = TAQ.get_offical_complete_nbbo(test, time, [symbol], nbbo_df, quote_df)
                        merged_df = TAQ.merge_trades_nbbo(test, trades_df, off_nnbo_df)

                        # merge day by day
                        old_df = pd.concat([old_df, merged_df], axis=0)
                    except:
                        pass
                # save our data
                gf.save_df(old_df, 'Data/TAQdata/data' + savedate + '_' + symbol + '.csv')

        # download tweets from elon musk's user profile
        if download_elon_twitter_data:
            print('Downloading elon twitter data')

            # copy and store start date
            twitter_start_date = start_date

            # get elon tweets for certain time period
            start_tweet = mktime((twitter_start_date + datetime.timedelta(hours=4)).timetuple())
            end_tweet = mktime((twitter_start_date + datetime.timedelta(hours=28)).timetuple())
            old_tweets = twitter.get_tweets('elonmusk', str(int(start_tweet)), str(int(end_tweet)))

            # retrieve for every date in our defined date-period
            while twitter_start_date < end_date:
                twitter_start_date += datetime.timedelta(days=1)

                # get elon tweets for certain time period
                start_tweet = mktime((twitter_start_date + datetime.timedelta(hours=4)).timetuple())
                end_tweet = mktime((twitter_start_date + datetime.timedelta(hours=28)).timetuple())
                tweets = twitter.get_tweets('elonmusk', str(int(start_tweet)), str(int(end_tweet)))

                # merge day by day
                old_tweets = pd.concat([old_tweets, tweets], axis=0)

            # remove duplicate tweets if any
            old_tweets.drop_duplicates(subset='Text', inplace=True)
            # convert time to datetime object
            old_tweets['Datetime'] = pd.to_datetime(old_tweets['Datetime']) - pd.Timedelta(hours=4)
            # save the raw tweet data-set to file
            gf.save_df(old_tweets, 'Data/Twitterdata/raw_data' + savedate + '_' + symbol + '.csv')

        # download twitter sentiment data about keywords
        if download_twitter_sentiment_data:
            print('Downloading twitter sentiment data')

            # copy and store start date
            twitter_start_date = start_date

            # get elon tweets for certain time period
            start_tweet = mktime((twitter_start_date + datetime.timedelta(hours=4)).timetuple())
            end_tweet = mktime((twitter_start_date + datetime.timedelta(hours=28)).timetuple())

            try:
                old_tweets = twitter.get_keyword_tweets(keywords, str(int(start_tweet)), str(int(end_tweet)), maxtweets)
            except:
                old_tweets = twitter.get_keyword_tweets(keywords, str(int(start_tweet)), str(int(end_tweet)), maxtweets)
                pass

            # retrieve for every date in our defined date-period
            while twitter_start_date < end_date:
                twitter_start_date += datetime.timedelta(days=1)
                print(twitter_start_date)
                # get elon tweets for certain time period
                start_tweet = mktime((twitter_start_date + datetime.timedelta(hours=4)).timetuple())
                end_tweet = mktime((twitter_start_date + datetime.timedelta(hours=28)).timetuple())
                try:
                    tweets = twitter.get_keyword_tweets(keywords, str(int(start_tweet)), str(int(end_tweet)), maxtweets)
                except:
                    tweets = twitter.get_keyword_tweets(keywords, str(int(start_tweet)), str(int(end_tweet)), maxtweets)
                    pass
                # merge day by day
                old_tweets = pd.concat([old_tweets, tweets], axis=0)

            # remove duplicate tweets if any
            old_tweets.drop_duplicates(subset='Text', inplace=True)
            # convert time to datetime object
            old_tweets['Datetime'] = pd.to_datetime(old_tweets['Datetime']) - pd.Timedelta(hours=4)
            # save the raw tweet data-set to file
            gf.save_df(old_tweets, 'Data/TwitterSentimentData/raw_data' + savedate + '_' + symbol + '.csv')

        if download_fama_risk_factors:
            print("start sql fama-french")

            # convert our datetime variables to the desired string format for sql query
            start_fama = start_date.strftime("%Y-%m-%d")
            end_fama = end_date.strftime("%Y-%m-%d")

            # the fama french sql request to WRDS!!!
            fama_factors = wrdsconn.raw_sql("""select date, mktrf, smb, hml, rf, umd 
                                    from ff.factors_daily
                                    where date>= %s
                                    and date<= %s""",
                                            params={(start_fama, end_fama)},
                                            date_cols=['date'])

            # sometimes the WRDS sql request outputs an empty dataframe, so then we have to retry the request
            # a line to thank WRDS for their services
            while len(fama_factors.index) < 10:
                fama_factors = wrdsconn.raw_sql("""select date, mktrf, smb, hml, rf, umd 
                                                    from ff.factors_daily
                                                    where date>= %s
                                                    and date<= %s""",
                                                params={start_fama, end_fama},
                                                date_cols=['date'])

            gf.save_df(fama_factors, 'Data/Fama_french/raw_data' + savedate + '_' + symbol + '.csv')

        # download finance data from Yahoo Finance
        if download_Yahoo_data:
            print('Downloading yahoo data')

            # our dates for yahoofinance data retrieval
            # extra days before the selected timeframe for rolling average for later 7 day volatility calculations
            start = start_date - datetime.timedelta(days=10)
            end = end_date

            # get our data
            Yahoo_data = yh.Getfinancedata(start, end, symbol)

            # save our data
            gf.save_df(Yahoo_data, 'Data/YahooFinanceData/raw_data' + savedate + '_' + symbol + '.csv')

        # download smp500 reference data
        if download_smp_500_data:
            print('Downloading smp500 data')

            # our dates for yahoofinance data retrieval
            # extra days before the selected timeframe for rolling average for later 7 day volatility calculations
            start = start_date - datetime.timedelta(days=10)  # extra 7 days for rolling average
            end = end_date

            # get our data
            smp500 = yh.Getfinancedata(start, end, '^GSPC')

            # save our data
            gf.save_df(smp500, 'Data/SMP500Data/raw_data' + savedate + '_' + symbol + '.csv')

        # process and clean elon musk's tweets
        if process_elon_twitter_data:
            print('Processing elon twitter data')

            # open the raw tweet data-set
            tweets = pd.read_csv('Data/Twitterdata/raw_data' + savedate + '_' + symbol + '.csv',
                                 parse_dates=['Datetime'])

            # rename Date column to timestamp
            tweets.rename(columns={'Date': 'Timestamp'}, inplace=True)

            # lowercase, remove twitter handles, delete hyperlinks, delete non-abc characters, remove spaces
            # remove stopwords + words such as rt, rts, retweet
            additional = ['rt', 'rts', 'retweet']
            swords = set().union(stopwords.words('english'), additional)
            tweets['processed_text'] = tweets['Text'].str.lower() \
                .str.replace('(&amp)', ' ', regex=True) \
                .str.replace('(@[a-z0-9]+)\w+', ' ', regex=True) \
                .str.replace('(http\S+)', ' ', regex=True) \
                .str.replace('([^0-9a-z$ \t])', ' ', regex=True) \
                .str.replace(' +', ' ', regex=True) \
                .apply(lambda x: [i for i in x.split() if not i in swords])

            # create instance of our sentiment analyser
            sia = vd.SentimentIntensityAnalyzer()
            tweets["processed_text"] = tweets["processed_text"].str.join(" ")

            # feed our tweets word by word in the sentiment analyser; and make a compound sentiment score for each tweet
            tweets['sentiment_score'] = [sia.polarity_scores(v)['compound']
                                         for v in tweets['processed_text']]

            # categorise our sentiment in pos / neutral / negative tweets
            tweets['sentiment_category'] = tweets.apply(lambda row: gf.categorize_sentiment(row['sentiment_score']),
                                                        axis=1)

            # # get the 500 tickers from the SP500
            # smp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            # smp500['Symbol'] = smp500['Symbol'].astype(str).str.lower()
            # smp500['Security'] = smp500['Security'].astype(str).str.lower()
            # smp500["$Symbol"] = '$' + smp500["Symbol"].astype(str)
            # smp500["Symbol"] = (' ' + smp500["Symbol"] + ' ').astype(str)
            # smp500_symbols = smp500['Symbol'].values.tolist()
            # smp500_dsymbols = smp500['$Symbol'].values.tolist()
            # smp500_names = smp500['Security'].values.tolist()
            #
            # # delete unnecessary columns
            # del smp500['SEC filings']
            # del smp500['GICS Sector']
            # del smp500['GICS Sub-Industry']
            # del smp500['Headquarters Location']
            # del smp500['Date first added']
            # del smp500['CIK']
            # del smp500['Founded']
            #
            # tweets['keyword'] = tweets['processed_text'].astype(str).str.findall(
            #     "|".join(['neuralink', 'tesla', 'spacex'])).apply(set).str.join(', ')

            # save our tweet dataset in csv format
            gf.save_df(tweets, 'Data/Twitterdata/processed_data' + savedate + '_' + symbol + '.csv')

        # process and clean keyword tweets
        if process_twitter_sentiment_data:
            print('Processing twitter sentiment data')

            # open the raw tweet data-set
            tweets = pd.read_csv('Data/TwitterSentimentData/raw_data' + savedate + '_' + symbol + '.csv',
                                 parse_dates=['Datetime'],
                                 lineterminator='\n')

            # rename Date column to timestamp
            tweets.rename(columns={'Date': 'Timestamp'}, inplace=True)
            tweets['Text'] = tweets['Text'].astype(str)

            # lowercase, remove twitter handles, delete hyperlinks, delete non-abc characters, remove spaces
            # remove stopwords + words such as rt, rts, retweet
            additional = ['rt', 'rts', 'retweet']
            swords = set().union(stopwords.words('english'), additional)
            tweets['processed_text'] = tweets['Text'].str.lower() \
                .str.replace('(&amp)', ' ', regex=True) \
                .str.replace('(@[a-z0-9]+)\w+', ' ', regex=True) \
                .str.replace('(http\S+)', ' ', regex=True) \
                .str.replace('([^0-9a-z$ \t])', ' ', regex=True) \
                .str.replace(' +', ' ', regex=True) \
                .apply(lambda x: [i for i in x.split() if not i in swords])

            # create instance of our sentiment analyser
            sia = vd.SentimentIntensityAnalyzer()
            tweets["processed_text"] = tweets["processed_text"].str.join(" ")

            # feed our tweets word by word in the sentiment analyser; and make a compound sentiment score for each tweet
            tweets['sentiment_score'] = [sia.polarity_scores(v)['compound']
                                         for v in tweets['processed_text']]

            # categorise our sentiment in pos / neutral / negative tweets
            tweets['sentiment_category'] = tweets.apply(lambda row: gf.categorize_sentiment(row['sentiment_score']),
                                                        axis=1)

            # # get the 500 tickers from the SP500
            # smp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            # smp500['Symbol'] = smp500['Symbol'].astype(str).str.lower()
            # smp500['Security'] = smp500['Security'].astype(str).str.lower()
            # smp500["$Symbol"] = '$' + smp500["Symbol"].astype(str)
            # smp500["Symbol"] = (' ' + smp500["Symbol"] + ' ').astype(str)
            # smp500_symbols = smp500['Symbol'].values.tolist()
            # smp500_dsymbols = smp500['$Symbol'].values.tolist()
            # smp500_names = smp500['Security'].values.tolist()
            #
            # # delete unnecessary columns
            # del smp500['SEC filings']
            # del smp500['GICS Sector']
            # del smp500['GICS Sub-Industry']
            # del smp500['Headquarters Location']
            # del smp500['Date first added']
            # del smp500['CIK']
            # del smp500['Founded']
            #
            # tweets['keyword'] = tweets['processed_text'].astype(str).str.findall(
            #     "|".join(['neuralink', 'tesla', 'spacex'])).apply(set).str.join(', ')

            # save our tweet dataset in csv format
            gf.save_df(tweets, 'Data/TwitterSentimentData/processed_data' + savedate + '_' + symbol + '.csv')

        # process and clean TAQ data
        if process_TAQ_data:
            print('Processing TAQ data')

            # only download TAQ data for dates within the data availability timeframe
            if end_date < datetime.datetime(2019, 12, 31):
                # read raw data from file
                taqdata = pd.read_csv('Data/TAQdata/data' + savedate + '_' + symbol + '.csv', parse_dates=['timestamp'])

                # calculate relative effective spread for each order
                taqdata['RES'] = 2 * abs(taqdata['price'] - taqdata['midpoint']) / taqdata['midpoint']

                # calculate orderbook imbalance
                taqdata['OIB'] = (taqdata['best_bid'] - taqdata['best_ask']) / (
                        taqdata['best_bid'] + taqdata['best_ask'])

                # create a new dataset only representing retail trades
                ret_taqdata = taqdata[taqdata['ex'] == 'D']
                ret_taqdata = ret_taqdata[ret_taqdata['symbol'] == 'TSLA']
                ret_taqdata = ret_taqdata[(ret_taqdata['BuySellBJZ'] == 1) | (ret_taqdata['BuySellBJZ'] == -1)]

                # create a new dataset only representing non-retail trades
                non_ret_taqdata = taqdata[taqdata['ex'] != 'D']
                non_ret_taqdata = non_ret_taqdata[non_ret_taqdata['symbol'] == 'TSLA']
                non_ret_taqdata = non_ret_taqdata[
                    (non_ret_taqdata['BuySellLRnotBJZ'] == 1) | (non_ret_taqdata['BuySellLRnotBJZ'] == -1)]

                # lets create the trading volume, sell trading volume and buy trading volume of the orders
                ret_taqdata.set_index(pd.to_datetime(ret_taqdata['timestamp']))
                ret_taqdata['ret_total_trading_volume'] = ret_taqdata.rolling('1D', on='timestamp')['size'].sum()
                ret_taqdata['ret_buy_trading_volume'] = \
                    ret_taqdata[ret_taqdata['BuySellBJZ'] == 1].rolling('1D', on='timestamp')['size'].sum()
                ret_taqdata['ret_sell_trading_volume'] = \
                    ret_taqdata[ret_taqdata['BuySellBJZ'] == -1].rolling('1D', on='timestamp')['size'].sum()

                non_ret_taqdata.set_index(pd.to_datetime(non_ret_taqdata['timestamp']))
                non_ret_taqdata['no_ret_total_trading_volume'] = non_ret_taqdata.rolling('1D', on='timestamp')[
                    'size'].sum()
                non_ret_taqdata['no_ret_buy_trading_volume'] = \
                    non_ret_taqdata[non_ret_taqdata['BuySellLRnotBJZ'] == 1].rolling('1D', on='timestamp')['size'].sum()
                non_ret_taqdata['no_ret_sell_trading_volume'] = \
                    non_ret_taqdata[non_ret_taqdata['BuySellLRnotBJZ'] == -1].rolling('1D', on='timestamp')[
                        'size'].sum()

                taqdata.set_index(pd.to_datetime(taqdata['timestamp']))
                taqdata['total_trading_volume'] = taqdata.rolling('1D', on='timestamp')['size'].sum()
                taqdata['buy_trading_volume'] = \
                    taqdata[(taqdata['BuySellLRnotBJZ'] == 1) | (taqdata['BuySellBJZ'] == 1)].rolling('1D',
                                                                                                      on='timestamp')[
                        'size'].sum()
                taqdata['sell_trading_volume'] = \
                    taqdata[(taqdata['BuySellLRnotBJZ'] == -1) | (taqdata['BuySellBJZ'] == 1)].rolling('1D',
                                                                                                       on='timestamp')[
                        'size'].sum()

                # save the data frames for further analysis later on
                gf.save_df(taqdata, 'Data/TAQdata/processed_data' + savedate + '_' + symbol + '.csv')
                gf.save_df(ret_taqdata, 'Data/TAQdata/retail_processed_data' + savedate + '_' + symbol + '.csv')
                gf.save_df(non_ret_taqdata, 'Data/TAQdata/non_retail_processed_data' + savedate + '_' + symbol + '.csv')

        # process and clean Yahoofinance data
        if process_Yahoo_data:
            print('Processing Yahoo data')

            # read the yahoo dataset
            Yahoo_data = pd.read_csv('Data/YahooFinanceData/raw_data' + savedate + '_' + symbol + '.csv',
                                     parse_dates=['Date'])

            # let's calculate the daily stock return of a retrieved stock
            Yahoo_data['D_log_return'] = (np.log(Yahoo_data['Close'] / Yahoo_data['Close'].shift(-1)))
            Yahoo_data['d_d_vol'] = Yahoo_data['D_log_return'].pct_change().rolling(7).std()

            # delete first 7 days necessary to initiate rolling avg
            Yahoo_data['Date'] = pd.to_datetime(Yahoo_data['Date'])
            Yahoo_data = Yahoo_data[~(Yahoo_data['Date'] < start_date)]

            # save the data frame for further analysis
            gf.save_df(Yahoo_data, 'Data/YahooFinanceData/processed_data' + savedate + '_' + symbol + '.csv')

        # process and clean smp500 benchmark data
        if process_smp500_data:
            print('Processing smp500 data')

            # read the smp500 dataset
            smp500 = pd.read_csv('Data/SMP500Data/raw_data' + savedate + '_' + symbol + '.csv', parse_dates=['Date'])

            # let's calculate the daily stock return of the smp500
            smp500['D_log_return'] = (np.log(smp500['Close'] / smp500['Close'].shift(-1)))
            smp500['d_d_vol'] = smp500['D_log_return'].pct_change().rolling(7).std()

            # delete first 7 days necessary to initiate rolling avg
            smp500['Date'] = pd.to_datetime(smp500['Date'])
            smp500 = smp500[~(smp500['Date'] < start_date)]

            # save the data frame for further analysis
            gf.save_df(smp500, 'Data/SMP500Data/processed_data' + savedate + '_' + symbol + '.csv')

        # generate the dataset used for the granger analysis study
        if generate_analysis_dataset:

            # only perform the granger analysis for events within the data availability interval
            if end_date < datetime.datetime(2019, 12, 31):
                print('Generating granger dataset')

                # read all of our data from the processed data files
                elontweets = pd.read_csv('Data/Twitterdata/processed_data' + savedate + '_' + symbol + '.csv',
                                         parse_dates=['Datetime'])
                RH_Data = pd.read_csv('Data/RHdata/popularity_export/' + symbol + '.csv', parse_dates=['timestamp'])
                senttweets = pd.read_csv('Data/TwitterSentimentData/processed_data' + savedate + '_' + symbol + '.csv',
                                         parse_dates=['Datetime'], low_memory=False, lineterminator='\n')

                taqdata_all = pd.read_csv('Data/TAQdata/processed_data' + savedate + '_' + symbol + '.csv',
                                          parse_dates=['timestamp'])
                taqdata_retail = pd.read_csv('Data/TAQdata/retail_processed_data' + savedate + '_' + symbol + '.csv',
                                             parse_dates=['timestamp'])
                taqdata_no_retail = pd.read_csv(
                    'Data/TAQdata/non_retail_processed_data' + savedate + '_' + symbol + '.csv',
                    parse_dates=['timestamp'])
                Yahoo_data = pd.read_csv('Data/YahooFinanceData/processed_data' + savedate + '_' + symbol + '.csv',
                                         parse_dates=['Date'])
                Fama_data = pd.read_csv('Data/Fama_french/raw_data' + savedate + '_' + symbol + '.csv',
                                        parse_dates=['date'])

                # only keep columns relevant data entries for our analysis, and rename some columns
                elontweets = elontweets[['Datetime', 'LikeCount', 'RetweetCount', 'sentiment_score']]
                elontweets.rename(columns={'sentiment_score': 'elon_sentiment'}, inplace=True)
                senttweets = senttweets[['Datetime', 'sentiment_score']]
                senttweets.rename(columns={'sentiment_score': 'twitter_sentiment'}, inplace=True)
                senttweets = senttweets.dropna()

                RH_Data['timestamp'] = pd.to_datetime(RH_Data['timestamp']) - pd.Timedelta(hours=4)
                RH_Data['Datetime'] = RH_Data['timestamp']
                RH_Data = RH_Data[['Datetime', 'users_holding']]

                taqdata_all['Datetime'] = taqdata_all['timestamp']
                taqdata_retail['Datetime'] = taqdata_retail['timestamp']
                taqdata_no_retail['Datetime'] = taqdata_no_retail['timestamp']

                Yahoo_data['Datetime'] = Yahoo_data['Date']
                Yahoo_data = Yahoo_data[['Datetime', 'Close', 'Volume', 'D_log_return', 'd_d_vol']]

                Fama_data['Datetime'] = Fama_data['date']
                Fama_data = Fama_data[['Datetime', 'smb', 'hml', 'rf', 'mktrf']]

                # only keep the columns relevant for our analysis
                taqdata_all = taqdata_all[
                    ['Datetime', 'price', 'RES', 'OIB', 'total_trading_volume', 'buy_trading_volume',
                     'sell_trading_volume']]
                taqdata_retail = taqdata_retail[
                    ['Datetime', 'ret_total_trading_volume', 'ret_buy_trading_volume', 'ret_sell_trading_volume']]
                taqdata_no_retail = taqdata_no_retail[
                    ['Datetime', 'no_ret_total_trading_volume', 'no_ret_buy_trading_volume',
                     'no_ret_sell_trading_volume']]

                # Let's merge our datasets in order to perform a granger analysis
                # and sort our data ascending
                # and round our dataframes to the nearest hour if applicable
                senttweets['Datetime'] = pd.to_datetime(senttweets['Datetime'], utc=True)
                elontweets['Datetime'] = pd.to_datetime(elontweets['Datetime'], utc=True)
                taqdata_all['Datetime'] = pd.to_datetime(taqdata_all['Datetime'], utc=True)
                taqdata_retail['Datetime'] = pd.to_datetime(taqdata_retail['Datetime'], utc=True)
                taqdata_no_retail['Datetime'] = pd.to_datetime(taqdata_no_retail['Datetime'], utc=True)
                RH_Data['Datetime'] = pd.to_datetime(RH_Data['Datetime'], utc=True)
                Yahoo_data['Datetime'] = pd.to_datetime(Yahoo_data['Datetime'], utc=True)
                Fama_data['Datetime'] = pd.to_datetime(Fama_data['Datetime'], utc=True)

                senttweets = senttweets.sort_values(by=['Datetime'], ascending=True)
                elontweets = elontweets.sort_values(by=['Datetime'], ascending=True)
                taqdata_all = taqdata_all.sort_values(by=['Datetime'], ascending=True)
                taqdata_retail = taqdata_retail.sort_values(by=['Datetime'], ascending=True)
                taqdata_no_retail = taqdata_no_retail.sort_values(by=['Datetime'], ascending=True)
                RH_Data = RH_Data.sort_values(by=['Datetime'], ascending=True)
                Yahoo_data = Yahoo_data.sort_values(by=['Datetime'], ascending=True)
                Fama_data = Fama_data.sort_values(by=['Datetime'], ascending=True)

                # lets merge and find nearest available datapoint, in case no datapoint leave empty
                df = pd.merge_asof(taqdata_all, taqdata_retail, on=['Datetime'], direction="nearest")  # retail data
                df1 = pd.merge_asof(df, taqdata_no_retail, on=['Datetime'], direction="nearest")  # no retail
                df2 = pd.merge_asof(df1, elontweets, on=['Datetime'], direction="forward")  # elontweet data
                df3 = pd.merge_asof(df2, RH_Data, on=['Datetime'], direction="forward")  # robinhood data
                df4 = pd.merge_asof(df3, senttweets, on=['Datetime'], direction="forward")  # twitter_sentiment data
                df5 = pd.merge_asof(df4, Yahoo_data, on=['Datetime'], direction="forward")  # yahoo data
                Granger_dataset = pd.merge_asof(df5, Fama_data, on=['Datetime'], direction="forward")  # fama factors

                # save the merged granger data-set
                Granger_dataset.set_index('Datetime')
                gf.save_df(Granger_dataset, 'Data/Grangerdata/merged_data' + savedate + '_' + symbol + '.csv')

        # perform the granger analysis
        if granger_analysis:
            print('Performing granger analysis')

            # only perform the granger analysis for events that lie within the data-availability interval
            if end_date < datetime.datetime(2019, 12, 31):

                # download - process - generate granger dataset done? lets continue!! else go back to start
                # read and parse the generated granger analysis data-set
                Granger_df = pd.read_csv('Data/Grangerdata/merged_data' + savedate + '_' + symbol + '.csv',
                                         parse_dates=['Datetime'],
                                         index_col='Datetime')

                # drop unnamed index column
                Granger_df.drop(Granger_df.filter(regex="Unnamed"), axis=1, inplace=True)

                # index to datetime
                Granger_df.index = pd.to_datetime(Granger_df.index)

                # resample our data to the 1h interval
                Granger_df = Granger_df.resample('1H').agg({'price': 'last',
                                                            'RES': 'last',
                                                            'OIB': 'last',
                                                            'total_trading_volume': 'sum',
                                                            'buy_trading_volume': 'sum',
                                                            'sell_trading_volume': 'sum',
                                                            'ret_total_trading_volume': 'sum',
                                                            'ret_buy_trading_volume': 'sum',
                                                            'ret_sell_trading_volume': 'sum',
                                                            'no_ret_total_trading_volume': 'sum',
                                                            'no_ret_buy_trading_volume': 'sum',
                                                            'no_ret_sell_trading_volume': 'sum',
                                                            'LikeCount': 'sum',
                                                            'RetweetCount': 'sum',
                                                            'elon_sentiment': 'mean',
                                                            'twitter_sentiment': 'mean',
                                                            'users_holding': 'last',
                                                            'Close': 'last',
                                                            'Volume': 'sum',
                                                            'D_log_return': 'last',
                                                            'd_d_vol': 'last',
                                                            'smb': 'last',
                                                            'hml': 'last',
                                                            'rf': 'last',
                                                            'mktrf': 'last'
                                                            })

                # finally calculate hourly returns
                Granger_df['price'] = Granger_df['price'].pct_change()

                # skip first row with NA
                Granger_df = Granger_df[1:]

                # drop na entries
                Granger_df = Granger_df.dropna()
                # Granger_df.fillna(method='ffill', inplace=True)

                # perform our fama-french regression over the complete time period
                # and retrieve our betas for the calculation of the expected rate of return
                # in turn we change the asset return to abnormal asset return

                X = Granger_df[['mktrf', 'smb', 'hml']]
                y = Granger_df['price'] - Granger_df['rf']
                X = sm.add_constant(X)
                ff_model = sm.OLS(y, X).fit()

                # print(ff_model.summary())
                intercept, b1, b2, b3 = ff_model.params

                Granger_df['price'] = Granger_df['price'] - (
                        b1 * Granger_df['mktrf'] + b2 * Granger_df['smb'] + b3 * Granger_df['hml'])

                # Cumulative abnormal return
                copy_granger_df = Granger_df
                copy_granger_df['price'] = (1 + copy_granger_df['price']).cumprod() - 1

                # standardise our dataset
                x = Granger_df.values  # returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(x)
                Granger_df = pd.DataFrame(x_scaled, columns=Granger_df.columns, index=Granger_df.index)
                Granger_df.fillna(method='ffill', inplace=True)

                # standardise our dataset
                x = copy_granger_df.values  # returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(x)
                copy_granger_df = pd.DataFrame(x_scaled, columns=copy_granger_df.columns, index=copy_granger_df.index)
                copy_granger_df.fillna(method='ffill', inplace=True)

                # for bookkeeping
                gf.save_df(copy_granger_df, 'Data/Grangerdata/granger_data' + savedate + '_' + symbol + '.csv')

                # decide for what variables we want to do the granger analysis
                Granger_df = Granger_df[
                    ['price', 'RES', 'OIB', 'total_trading_volume', 'ret_total_trading_volume', 'LikeCount',
                     'RetweetCount',
                     'elon_sentiment', 'twitter_sentiment', 'users_holding', 'D_log_return', 'd_d_vol']]

                # perform the adf test for stationarity, and if not stationary,
                # automatically differentiate and check again
                newlag = 1
                for column in Granger_df:
                    [Granger_df[column], lag] = gf.adf_test(Granger_df[column])
                    if lag > newlag:
                        newlag = lag

                # Split our dataframe in data from the month before the discrete event
                # and data from the month after the discrete event
                # 2 weeks before and 2 weeks after, 5 trading days a week; 10 trading days
                # 7 trading hours per day; 70 datapoints in total per analysis

                Granger_df_before = Granger_df[~(Granger_df.index.date > discrete_date.date())].tail(70)
                Granger_df_after = Granger_df[~(Granger_df.index.date <= discrete_date.date())].head(70)

                # perform the two granger analyses with a maximum lag of 7hrs, equalling 1 trading day
                Granger_matrix_before = gf.granger_causality_matrix(Granger_df_before,
                                                                    variables=Granger_df_before.columns,
                                                                    lag=7)
                Granger_matrix_after = gf.granger_causality_matrix(Granger_df_after, variables=Granger_df_after.columns,
                                                                   lag=7)

                # save the granger analysis results
                gf.save_df(Granger_matrix_before,
                           'AnalysisResults/Granger/granger_result_before_' + savedate + '_' + symbol + '.csv')
                gf.save_df(Granger_matrix_after,
                           'AnalysisResults/Granger/granger_result_after_' + savedate + '_' + symbol + '.csv')
                # # let's make a plot with the data subjected to our analysis
                # fig, axs = plt.subplots(4)
                # fig.suptitle('Discrete information event timeline')
                # taqdata.plot(ax=axs[0], x='Datetime', y='price')
                # elontweets.plot(ax=axs[1], x='Datetime', y='sentiment_score')
                # elontweets.plot(ax=axs[2], x='Datetime', y='RetweetCount')
                # RH_Data.plot(ax=axs[3], x='Datetime', y='users_holding')
                #
                # axs[0].set_title('Stock Price')
                # axs[1].set_title('Elontweet Sentiment')
                # axs[2].set_title('RetweetCount')
                # axs[3].set_title('Users Holding')
                #
                # axs[0].set_xlim([start_date, end_date])
                # axs[1].set_xlim([start_date, end_date])
                # axs[2].set_xlim([start_date, end_date])
                # axs[3].set_xlim([start_date, end_date])
                # plt.xticks(rotation=45)
                # plt.draw()

        # perform the cumulative return analysis
        if cumulative_return_analysis:
            print('Performing cumulative return analysis')

            # read our stock data
            stock_data = pd.read_csv('Data/YahooFinanceData/processed_data' + savedate + '_' + symbol + '.csv',
                                     parse_dates=['Date'])
            stock_data['Datetime'] = stock_data['Date']
            stock_data = stock_data[['Datetime', 'Close']]
            stock_data.rename(columns={'Close': 'Close_stock'}, inplace=True)
            stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'], utc=True)
            stock_data = stock_data.sort_values(by=['Datetime'], ascending=True)

            # read our reference smp500 data
            smp500 = pd.read_csv('Data/SMP500Data/processed_data' + savedate + '_' + symbol + '.csv',
                                 parse_dates=['Date'])
            smp500['Datetime'] = smp500['Date']
            smp500 = smp500[['Datetime', 'Close']]
            smp500.rename(columns={'Close': 'Close_smp500'}, inplace=True)
            smp500['Datetime'] = pd.to_datetime(smp500['Datetime'], utc=True)
            smp500 = smp500.sort_values(by=['Datetime'], ascending=True)

            # merge our dataframes
            merged_cumulative = pd.merge_asof(stock_data, smp500, on=['Datetime'], direction="nearest")  # retail data

            # our daily stock returns
            merged_cumulative['Close_smp500_daily'] = merged_cumulative['Close_smp500'].pct_change()
            merged_cumulative['Close_stock_daily'] = merged_cumulative['Close_stock'].pct_change()

            # skip first row with NA
            merged_cumulative = merged_cumulative[1:]

            # Calculate the cumulative daily returns
            merged_cumulative['Close_smp500_cumulative_daily'] = (1 + merged_cumulative[
                'Close_smp500_daily']).cumprod() - 1
            merged_cumulative['Close_stock_cumulative_daily'] = (1 + merged_cumulative[
                'Close_stock_daily']).cumprod() - 1
            merged_cumulative['Close_delta_cumulative_daily'] = merged_cumulative['Close_stock_cumulative_daily'] - \
                                                                merged_cumulative['Close_smp500_cumulative_daily']
            merged_cumulative = merged_cumulative.reset_index()
            gf.save_df(merged_cumulative,
                       'AnalysisResults/CumulativeReturn/cumulative_return' + savedate + '_' + symbol + '.csv')

        # perform the cumulative return analysis
        if fama_cumulative_return_analysis:
            print('Performing fama-french cumulative return analysis')

            # read our stock data
            stock_data = pd.read_csv('Data/YahooFinanceData/processed_data' + savedate + '_' + symbol + '.csv',
                                     parse_dates=['Date'])
            stock_data['Datetime'] = stock_data['Date']
            stock_data = stock_data[['Datetime', 'Close']]
            stock_data.rename(columns={'Close': 'Close_stock'}, inplace=True)
            stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'], utc=True)
            stock_data = stock_data.sort_values(by=['Datetime'], ascending=True)

            # read our reference fama-macbeth data
            fama = pd.read_csv('Data/Fama_french/raw_data' + savedate + '_' + symbol + '.csv',
                               parse_dates=['date'])
            fama['Datetime'] = fama['date']
            fama = fama[['Datetime', 'smb', 'hml', 'rf', 'mktrf']]
            fama['Datetime'] = pd.to_datetime(fama['Datetime'], utc=True)
            fama = fama.sort_values(by=['Datetime'], ascending=True)

            # merge our dataframes
            merged = pd.merge_asof(fama, stock_data, on=['Datetime'], direction="nearest")  # retail data

            # our daily stock returns
            merged['Close_stock_daily'] = merged['Close_stock'].pct_change()
            # skip first row with NA
            merged = merged[1:]

            # perform our fama-macbeth regression over the complete time period
            # and retrieve our betas for the calculation of the expected rate of return
            X = merged[['mktrf', 'smb', 'hml']]
            y = merged['Close_stock_daily'] - merged['rf']
            X = sm.add_constant(X)
            ff_model = sm.OLS(y, X).fit()
            print(ff_model.summary())
            intercept, b1, b2, b3 = ff_model.params

            merged['abnormal_return'] = merged['Close_stock_daily'] - (
                    b1 * merged['mktrf'] + b2 * merged['smb'] + b3 * merged['hml'])

            # Calculate the cumulative daily returns
            merged['Close_abnormal_cumulative_daily'] = (1 + merged[
                'abnormal_return']).cumprod() - 1

            merged = merged.reset_index()
            gf.save_df(merged,
                       'AnalysisResults/CumulativeReturn_fama/cumulative_return' + savedate + '_' + symbol + '.csv')

        # perform the simple regression analysis
        if simple_regression_analysis:
            # only perform the simple regression for events within the data availability interval
            if end_date < datetime.datetime(2019, 12, 31):
                # arrays for individual regression results
                regression_results_ind_before = []
                regression_results_ind_after = []

                # we use the dataset we already made for the granger analysis
                Regression_df = pd.read_csv('Data/Grangerdata/granger_data' + savedate + '_' + symbol + '.csv',
                                            parse_dates=['Datetime'],
                                            index_col='Datetime')
                # only keep columns we want
                Regression_df = Regression_df[
                    ['price', 'RES', 'OIB', 'total_trading_volume', 'ret_total_trading_volume', 'LikeCount',
                     'RetweetCount', 'elon_sentiment', 'twitter_sentiment', 'users_holding', 'D_log_return',
                     'd_d_vol']]

                Regression_df_before = Regression_df[~(Regression_df.index.date > discrete_date.date())].tail(70)
                Regression_df_after = Regression_df[~(Regression_df.index.date <= discrete_date.date())].head(70)

                # full regression model
                xb = Regression_df_before[
                    ['RES', 'OIB', 'total_trading_volume', 'ret_total_trading_volume', 'LikeCount',
                     'RetweetCount', 'elon_sentiment', 'twitter_sentiment', 'users_holding', 'D_log_return',
                     'd_d_vol']]

                yb = Regression_df_before['price']

                xa = Regression_df_after[
                    ['RES', 'OIB', 'total_trading_volume', 'ret_total_trading_volume', 'LikeCount',
                     'RetweetCount', 'elon_sentiment', 'twitter_sentiment', 'users_holding', 'D_log_return',
                     'd_d_vol']]

                ya = Regression_df_after['price']

                # individual regression before
                for column in xb:
                    Xb0 = sm.add_constant(xb[column])  # adding a constant
                    model_b0 = sm.OLS(yb, Xb0).fit()
                    predictions_b = model_b0.predict(Xb0)
                    print_model_b = model_b0.summary()
                    regression_results_ind_before.append(model_b0)
                    print(print_model_b)
                stargazerindb = Stargazer(regression_results_ind_before)

                # individual regression after
                for column in xa:
                    Xb0 = sm.add_constant(xa[column])  # adding a constant
                    model_b0 = sm.OLS(ya, Xb0).fit()
                    predictions_b = model_b0.predict(Xb0)
                    print_model_b = model_b0.summary()
                    regression_results_ind_after.append(model_b0)
                    print(print_model_b)
                stargazerinda = Stargazer(regression_results_ind_after)

                print("------------------------------------")
                print('REGRESSION BEFORE INDIVIDUAL')
                print(stargazerindb.render_latex())
                print("------------------------------------")
                print('REGRESSION AFTER INDIVIDUAL')
                print(stargazerinda.render_latex())

        # perform the multivariate regression analysis
        if multivariate_regression_analysis:

            # only perform the multivariate regression for events within the data availability interval
            if end_date < datetime.datetime(2019, 12, 31):

                # we use the dataset we already made for the granger analysis
                Regression_df = pd.read_csv('Data/Grangerdata/granger_data' + savedate + '_' + symbol + '.csv',
                                            parse_dates=['Datetime'],
                                            index_col='Datetime')

                # only keep columns we want
                Regression_df = Regression_df[
                    ['price', 'RES', 'OIB', 'total_trading_volume', 'ret_total_trading_volume', 'LikeCount',
                     'RetweetCount', 'elon_sentiment', 'twitter_sentiment', 'users_holding', 'D_log_return',
                     'd_d_vol']]

                # we split it the same way as for the granger analysis
                # 2 weeks before and 2 weeks after, 5 trading days a week; 10 trading days
                # 7 trading hours per day; 70 datapoints in total
                # then we add our dummy variable and concat the df back together
                Regression_df_before = Regression_df[~(Regression_df.index.date > discrete_date.date())].tail(70)
                Regression_df_before['event'] = 0
                Regression_df_after = Regression_df[~(Regression_df.index.date <= discrete_date.date())].head(70)
                Regression_df_after['event'] = 1

                # concat back together with the dummy variable present
                Regression_df_full = pd.concat([Regression_df_before, Regression_df_after], axis=0)

                # full regression model
                x = Regression_df_full[
                    ['RES', 'OIB', 'total_trading_volume', 'ret_total_trading_volume', 'LikeCount',
                     'RetweetCount', 'elon_sentiment', 'twitter_sentiment', 'users_holding', 'D_log_return',
                     'd_d_vol', 'event']]

                y = Regression_df_full['price']

                # let's check multicollinearity between our variables and drop variables if necessary
                # VIF dataframe
                print("------------------------------------")
                print('VIF ANALYSIS')

                vif_data = pd.DataFrame()
                vif_data["feature"] = x.columns

                # calculating VIF for each feature
                vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                                   for i in range(len(x.columns))]

                if event_id == 0:
                    vif_tables['feature'] = (vif_data['feature'])
                vif_tables['VIF event ' + str(event_id)] = (vif_data['VIF'])

                # regression model without multicollinearity vars
                data = x[['RES', 'ret_total_trading_volume', 'RetweetCount', 'elon_sentiment',
                          'twitter_sentiment', 'users_holding', 'd_d_vol', 'event']]
                data['price'] = Regression_df_full['price']

                # Now we perform our multivariate regression analysis
                # with the interaction of our dummy variables included
                basic_formula = 'price ~ ( RES + ret_total_trading_volume + d_d_vol + users_holding)'
                no_interaction_formula = 'price ~ (RES + ret_total_trading_volume ' \
                                         '+ RetweetCount + elon_sentiment + twitter_sentiment + users_holding + ' \
                                         'd_d_vol) '
                full_formula = 'price ~   event*(RES + ret_total_trading_volume ' \
                               '+ RetweetCount + elon_sentiment + twitter_sentiment + users_holding + d_d_vol)'

                mult_regressions = []
                model_basic = sm.formula.ols(basic_formula, data=data)
                model_full_no_interact = sm.formula.ols(no_interaction_formula, data=data)
                model_full = sm.formula.ols(full_formula, data=data)

                model_basic = model_basic.fit()
                model_no_inter_full = model_full_no_interact.fit()
                model_full = model_full.fit()

                print_model_basic = model_basic.summary()
                print_model_full_no_interact = model_no_inter_full.summary()
                print_model_full = model_full.summary()
                print(print_model_basic)
                print(print_model_full_no_interact)
                print(print_model_full)

                full_regressions.append(model_basic)
                full_regressions.append(model_no_inter_full)
                full_regressions.append(model_full)
                mult_regressions.append(model_basic)
                mult_regressions.append(model_no_inter_full)
                mult_regressions.append(model_full)
                fullstargazer = Stargazer(mult_regressions)
                print(fullstargazer.render_latex())
                parameters_basic = model_basic.params
                parameters_full_no_interact = model_no_inter_full.params
                parameters_full = model_full.params

                parameters_basic['price'] = np.mean(data['price'], 0)
                parameters_full_no_interact['price'] = np.mean(data['price'], 0)
                parameters_full['price'] = np.mean(data['price'], 0)

                parameters_basic['pos_neg'] = events[event_id][3]
                parameters_full_no_interact['pos_neg'] = events[event_id][3]
                parameters_full['pos_neg'] = events[event_id][3]

                fama_df_basic["event " + str(event_id)] = parameters_basic
                fama_df_full_no_interact["event " + str(event_id)] = parameters_full_no_interact
                fama_df_full["event " + str(event_id)] = parameters_full

        if simple_multivariate_regression_analysis:
            print('Performing simple multivariate regression analysis')

            # read all of our data from the processed data files
            elontweets = pd.read_csv('Data/Twitterdata/processed_data' + savedate + '_' + symbol + '.csv',
                                     parse_dates=['Datetime'])
            senttweets = pd.read_csv('Data/TwitterSentimentData/processed_data' + savedate + '_' + symbol + '.csv',
                                     parse_dates=['Datetime'], low_memory=False, lineterminator='\n')
            Yahoo_data = pd.read_csv('Data/YahooFinanceData/processed_data' + savedate + '_' + symbol + '.csv',
                                     parse_dates=['Date'])
            Fama_data = pd.read_csv('Data/Fama_french/raw_data' + savedate + '_' + symbol + '.csv',
                                    parse_dates=['date'])

            # only keep columns relevant data entries for our analysis, and rename some columns
            elontweets = elontweets[['Datetime', 'LikeCount', 'RetweetCount', 'sentiment_score']]
            elontweets.rename(columns={'sentiment_score': 'elon_sentiment'}, inplace=True)
            senttweets = senttweets[['Datetime', 'sentiment_score']]
            senttweets.rename(columns={'sentiment_score': 'twitter_sentiment'}, inplace=True)
            senttweets = senttweets.dropna()

            Yahoo_data['Datetime'] = Yahoo_data['Date']
            Yahoo_data = Yahoo_data[['Datetime', 'Close', 'Volume', 'D_log_return', 'd_d_vol']]

            Fama_data['Datetime'] = Fama_data['date']
            Fama_data = Fama_data[['Datetime', 'smb', 'hml', 'rf', 'mktrf']]

            # Let's merge our datasets in order to perform a granger analysis
            # and sort our data ascending
            # and round our dataframes to the nearest hour if applicable
            senttweets['Datetime'] = pd.to_datetime(senttweets['Datetime'], utc=True)
            elontweets['Datetime'] = pd.to_datetime(elontweets['Datetime'], utc=True)
            Yahoo_data['Datetime'] = pd.to_datetime(Yahoo_data['Datetime'], utc=True)
            Fama_data['Datetime'] = pd.to_datetime(Fama_data['Datetime'], utc=True)

            senttweets = senttweets.sort_values(by=['Datetime'], ascending=True)
            elontweets = elontweets.sort_values(by=['Datetime'], ascending=True)
            Yahoo_data = Yahoo_data.sort_values(by=['Datetime'], ascending=True)
            Fama_data = Fama_data.sort_values(by=['Datetime'], ascending=True)

            # lets merge and find nearest available datapoint, in case no datapoint leave empty
            df4 = pd.merge_asof(elontweets, senttweets, on=['Datetime'], direction="forward")  # twitter_sentiment data
            df5 = pd.merge_asof(df4, Yahoo_data, on=['Datetime'], direction="forward")  # yahoo data
            r_data = pd.merge_asof(df5, Fama_data, on=['Datetime'], direction="forward")  # fama factors
            r_data = r_data.dropna()

            r_data['Datetime'] = pd.to_datetime(r_data['Datetime'], utc=True)
            r_data.reset_index(inplace=True)

            # drop unnamed index column
            r_data.set_index(r_data['Datetime'], inplace=True)

            # resample our data to the 1h interval
            r_data = r_data.resample('1D').agg({'RetweetCount': 'sum',
                                                'elon_sentiment': 'mean',
                                                'twitter_sentiment': 'mean',
                                                'Close': 'last',
                                                'Volume': 'sum',
                                                'd_d_vol': 'last',
                                                'smb': 'last',
                                                'hml': 'last',
                                                'rf': 'last',
                                                'mktrf': 'last'
                                                })
            r_data = r_data.dropna()
            # finally calculate hourly returns
            r_data['Close'] = r_data['Close'].pct_change()

            # skip first row with NA
            r_data = r_data[1:]

            # perform our fama-french regression over the complete time period
            # and retrieve our betas for the calculation of the expected rate of return
            # in turn we change the asset return to abnormal asset return

            X = r_data[['mktrf', 'smb', 'hml']]
            y = r_data['Close'] - r_data['rf']
            X = sm.add_constant(X)
            ff_model = sm.OLS(y, X).fit()

            # print(ff_model.summary())
            intercept, b1, b2, b3 = ff_model.params

            r_data['price'] = r_data['Close'] - (
                    b1 * r_data['mktrf'] + b2 * r_data['smb'] + b3 * r_data['hml'])

            # Cumulative abnormal return
            r_data['price'] = (1 + r_data['price']).cumprod() - 1

            # standardise our dataset
            x = r_data.values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            r_data = pd.DataFrame(x_scaled, columns=r_data.columns, index=r_data.index)
            r_data.fillna(method='ffill', inplace=True)

            # we split it the same way as for the granger analysis
            # 2 weeks before and 2 weeks after, 5 trading days a week; 10 trading days
            # 7 trading hours per day; 70 datapoints in total
            # then we add our dummy variable and concat the df back together
            Regression_df_before = r_data[~(r_data.index.date >= discrete_date.date())].tail(70)
            Regression_df_before['event'] = 0
            Regression_df_after = r_data[~(r_data.index.date < discrete_date.date())].head(70)
            Regression_df_after['event'] = 1

            # concat back together with the dummy variable present
            Regression_df_full = pd.concat([Regression_df_before, Regression_df_after], axis=0)

            # Now we perform our multivariate regression analysis
            # with the interaction of our dummy variables included
            full_formula = 'price ~ event*(Volume + elon_sentiment)'

            model_full = sm.formula.ols(full_formula, data=Regression_df_full)
            model_full = model_full.fit()
            print_model_full = model_full.summary()
            full_regressions.append(model_full)
            print(print_model_full)
            parameters_full = model_full.params
            parameters_full['price'] = np.mean(Regression_df_full['price'], 0)
            parameters_full['pos_neg'] = events[event_id][3]
            simple_fama_df_full["event " + str(event_id)] = parameters_full
        # export the granger analysis results to latex table
        if export_granger_analysis_latex:
            if end_date < datetime.datetime(2019, 12, 31):
                print("BEFORE")
                granger_df = pd.read_csv(
                    'AnalysisResults/Granger/granger_result_before_' + savedate + '_' + symbol + '.csv')
                granger_df = granger_df[
                    ['Unnamed: 0', 'price_x', 'RES_x', 'OIB_x', 'total_trading_volume_x', 'ret_total_trading_volume_x',
                     'LikeCount_x',
                     'RetweetCount_x', 'elon_sentiment_x', 'twitter_sentiment_x',
                     'users_holding_x', 'D_log_return_x', 'd_d_vol_x']]

                print(granger_df.to_latex())

                print("AFTER")
                granger_df = pd.read_csv(
                    'AnalysisResults/Granger/granger_result_after_' + savedate + '_' + symbol + '.csv')
                granger_df = granger_df[
                    ['Unnamed: 0', 'price_x', 'RES_x', 'OIB_x', 'total_trading_volume_x', 'ret_total_trading_volume_x',
                     'LikeCount_x',
                     'RetweetCount_x', 'elon_sentiment_x', 'twitter_sentiment_x',
                     'users_holding_x', 'D_log_return_x', 'd_d_vol_x']]

                print(granger_df.to_latex())

    if simple_fama_cross_sectional_regression_analysis:
        print("Performing simple cross-sectional analysis")

        # our results from the multivariate regressions
        cross_df_full = simple_fama_df_full.T

        # our cross-sectional regression model
        full_formula = 'price ~   pos_neg*(event + Volume  ' \
                       ' + elon_sentiment  + event:Volume  ' \
                       ' + event:elon_sentiment )'

        # training and fitting the models
        model_full = sm.formula.ols(full_formula, data=cross_df_full)
        model_fullf = model_full.fit()
        print_model_full = model_fullf.summary()
        print(print_model_full)

        crossstargazer = Stargazer([model_fullf])
        print(crossstargazer.render_latex())
        print(cross_df_full.to_latex())
    if fama_cross_sectional_regression_analysis:
        print("Performing simple cross-sectional analysis")

        # our results from the multivariate regressions
        cross_df_basic = fama_df_basic.T
        cross_df_full_no_interact = fama_df_full_no_interact.T
        cross_df_full = fama_df_full.T

        # our cross-sectional regression model
        basic_formula = 'price ~ pos_neg*(RES + ret_total_trading_volume + d_d_vol + users_holding)'
        no_interaction_formula = 'price ~ pos_neg*(RES + ret_total_trading_volume ' \
                                 '+ RetweetCount + elon_sentiment + twitter_sentiment + users_holding + ' \
                                 'd_d_vol) '
        full_formula = 'price ~   pos_neg*event*(RES + ret_total_trading_volume ' \
                       '+ RetweetCount + elon_sentiment + twitter_sentiment + users_holding + d_d_vol)'

        # training and fitting the models
        model_basic = sm.formula.ols(basic_formula, data=cross_df_basic)
        model_full_no_interact = sm.formula.ols(no_interaction_formula, data=cross_df_full_no_interact)
        model_full = sm.formula.ols(full_formula, data=cross_df_full)
        model_basic = model_basic.fit()
        model_no_inter_full = model_full_no_interact.fit()
        model_full = model_full.fit()

        print_model_basic = model_basic.summary()
        print_model_full_no_interact = model_no_inter_full.summary()
        print_model_full = model_full.summary()
        print(print_model_basic)
        print(print_model_full_no_interact)
        print(print_model_full)

        full_regressions_fama.append(model_basic)
        full_regressions_fama.append(model_no_inter_full)
        full_regressions_fama.append(model_full)

    # export discrete event excel to latex table
    if export_discrete_events_latex:
        excel = pd.read_excel('Data/Discrete_events.xlsx')
        excel['Time'] = pd.to_datetime(excel['Time'])
        excel['Date'] = pd.to_datetime(excel['Date'])
        excel['Time'] = excel['Time'].dt.time
        excel['Date'] = excel['Date'].dt.date
        print(excel[['Date', 'Time', 'Tweet', 'Context Explanation', 'Result']].to_latex())

    # plot TAQ data for visualisation
    if plot_TAQ_data:
        taq_data = gf.get_df('Data/TAQdata/data.csv')
        print(taq_data)

    # plot the cumulative return data for visualisation
    if plot_cumulative_return:
        all_cum_return = pd.DataFrame()
        all_cum_return_norm = pd.DataFrame()
        for event_id in range(0, 15):
            # discrete event information
            symbol = events[event_id][1]  # ticker symbol
            date = events[event_id][0]  # 01/04/18' '14/05/18'  '07/08/18'
            savedate = date.replace("/", "_")  # replace forward slashes with underscore for saving
            df = pd.read_csv('AnalysisResults/CumulativeReturn/cumulative_return' + savedate + '_' + symbol + '.csv')
            df = df[['Close_delta_cumulative_daily', 'Datetime']]
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            index = df.index[df['Datetime'] == datetime.datetime.strptime(date, '%d/%m/%y')]

            while index.size == 0:
                if type(date) == str:
                    date = datetime.datetime.strptime(date, '%d/%m/%y') + datetime.timedelta(days=1)
                else:
                    date = date + datetime.timedelta(days=1)
                index = df.index[df['Datetime'] == date]
            index = index.item()
            df = df[['Close_delta_cumulative_daily']] * 100
            df = df[index - 8: index + 7]
            df = df.reset_index()
            df_norm = df.copy()

            df.rename(columns={'Close_delta_cumulative_daily': 'index ' + str(event_id)}, inplace=True)
            all_cum_return['ID ' + str(event_id) + ': $' + events[event_id][1]] = df['index ' + str(event_id)]
            df_norm['Close_delta_cumulative_daily'] = (df_norm['Close_delta_cumulative_daily'] - df_norm[
                'Close_delta_cumulative_daily'].mean()) / df_norm['Close_delta_cumulative_daily'].std()
            df_norm.rename(columns={'Close_delta_cumulative_daily': 'index ' + str(event_id)}, inplace=True)
            all_cum_return_norm['ID ' + str(event_id) + ': $' + events[event_id][1]] = df_norm['index ' + str(event_id)]

            # gamestop too
        all_cum_return_norm['ID 4: $' + events[4][1]] = all_cum_return_norm['ID 4: $' + events[4][1]] / 100
        all_cum_return['ID 4: $' + events[4][1]] = all_cum_return['ID 4: $' + events[4][1]] / 100
        all_cum_return.rename(columns={'ID 4: $' + events[4][1]: 'ID 4: $' + events[4][1] + '/100'}, inplace=True)
        all_cum_return_norm.rename(columns={'ID 4: $' + events[4][1]: 'ID 4: $' + events[4][1] + '/100'}, inplace=True)

        ax = all_cum_return.plot(use_index=True, xticks=range(0, 15))
        plt.axvline(x=7, color="green", label="event")
        ax.set_xticklabels([-7, -6, -5, -4, -3, -2, -1, -0, 1, 2, 3, 4, 5, 6, 7])
        plt.xlabel('Days since event [-]')
        plt.ylabel('CSR [%]')
        plt.title('CSR w.r.t s&p500 index')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.draw()
        print(all_cum_return.round(3).iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].to_latex())
        print(all_cum_return.round(3).iloc[:, [8, 9, 10, 11, 12, 13, 14]].to_latex())
        print(all_cum_return.round(3).to_latex())

    # plot the cumulative return data for visualisation
    if plot_cumulative_return_fama:
        all_cum_return = pd.DataFrame()
        all_cum_return_norm = pd.DataFrame()
        for event_id in range(0, 15):
            # discrete event information
            symbol = events[event_id][1]  # ticker symbol
            date = events[event_id][0]  # 01/04/18' '14/05/18'  '07/08/18'
            savedate = date.replace("/", "_")  # replace forward slashes with underscore for saving
            df = pd.read_csv(
                'AnalysisResults/CumulativeReturn_fama/cumulative_return' + savedate + '_' + symbol + '.csv')
            df = df[['Close_abnormal_cumulative_daily', 'Datetime']]
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            index = df.index[df['Datetime'] == datetime.datetime.strptime(date, '%d/%m/%y')]

            while index.size == 0:
                if type(date) == str:
                    date = datetime.datetime.strptime(date, '%d/%m/%y') + datetime.timedelta(days=1)
                else:
                    date = date + datetime.timedelta(days=1)
                index = df.index[df['Datetime'] == date]
            index = index.item()
            df = df[['Close_abnormal_cumulative_daily']] * 100
            df = df[index - 8: index + 7]
            df = df.reset_index()
            df_norm = df.copy()

            df.rename(columns={'Close_abnormal_cumulative_daily': 'index ' + str(event_id)}, inplace=True)
            all_cum_return['ID ' + str(event_id) + ': $' + events[event_id][1]] = df['index ' + str(event_id)]
            df_norm['Close_abnormal_cumulative_daily'] = (df_norm['Close_abnormal_cumulative_daily'] - df_norm[
                'Close_abnormal_cumulative_daily'].mean()) / df_norm['Close_abnormal_cumulative_daily'].std()
            df_norm.rename(columns={'Close_abnormal_cumulative_daily': 'index ' + str(event_id)}, inplace=True)
            all_cum_return_norm['ID ' + str(event_id) + ': $' + events[event_id][1]] = df_norm['index ' + str(event_id)]

            # gamestop too
        all_cum_return_norm['ID 4: $' + events[4][1]] = all_cum_return_norm['ID 4: $' + events[4][1]] / 100
        all_cum_return['ID 4: $' + events[4][1]] = all_cum_return['ID 4: $' + events[4][1]] / 100
        all_cum_return.rename(columns={'ID 4: $' + events[4][1]: 'ID 4: $' + events[4][1] + '/100'}, inplace=True)
        all_cum_return_norm.rename(columns={'ID 4: $' + events[4][1]: 'ID 4: $' + events[4][1] + '/100'}, inplace=True)

        ax = all_cum_return.plot(use_index=True, xticks=range(0, 15))
        plt.axvline(x=7, color="green", label="event")
        ax.set_xticklabels([-7, -6, -5, -4, -3, -2, -1, -0, 1, 2, 3, 4, 5, 6, 7])
        plt.xlabel('Days since event [-]')
        plt.ylabel('CASR [%]')
        plt.title('CASR w.r.t fama-3 factor model')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.draw()
        print(all_cum_return.round(3).iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].to_latex())
        print(all_cum_return.round(3).iloc[:, [8, 9, 10, 11, 12, 13, 14]].to_latex())
        print(all_cum_return.round(3).to_latex())

    # plot data on elon musk's twitter behaviour
    if plot_elon_twitter_behaviour:
        # read the twitter data from the processed data file
        elontweets = pd.read_csv('Data/Twitterdata/processed_data' + savedate + '_' + symbol + '.csv',
                                 parse_dates=['Datetime'])

        # calculate cumulative number of tweets over time and add them to a df
        elontweets['cumulative_n_tweets'] = 1
        elontweets['cumulative_n_tweets'] = elontweets['cumulative_n_tweets'].cumsum()
        elontweets['Datetime'] = elontweets['Datetime'] - datetime.timedelta(hours=4)
        # let's make a plot with the data subjected to our analysis
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex='all')
        # fig.suptitle('Elon Musk twitter behavour')
        elontweets.plot(ax=ax1, x='Datetime', y='cumulative_n_tweets', ylabel='n tweets')
        elontweets.plot(ax=ax2, x='Datetime', y='RetweetCount', ylabel='RetweetCount')
        elontweets.plot(ax=ax3, x='Datetime', y='LikeCount', ylabel='LikeCount', xlabel='Date')

        #  ax1.set_title('Cumulative number of tweets')
        # ax2.set_title('Number of retweets/tweet')
        # ax3.set_title('Number of likes/tweet')
        # ax1.set_yscale('log')
        # ax2.set_yscale('log')
        # ax3.set_yscale('log')
        ax1.legend(["Cumulative number of tweets"])
        ax2.legend(["Number of retweets/tweet"])
        ax3.legend(["Number of likes/tweet"])
        # plt.xticks(rotation=0)
        # plt.tight_layout()
        if export_tikz:
            mtikz.save("Plots/H1/Elon_behaviour_1.tex")
        plt.draw()

        # sort by hour
        tweets_by_hour = elontweets['Datetime'].dt.hour.value_counts().sort_index()
        tweets_by_year = elontweets['Datetime'].dt.year.value_counts().sort_index()
        # lets create a graph showing how many tweets of all of elon's tweets are at what time in the day
        # create bar plot
        fig, axs = plt.subplots(nrows=3, sharex='all')
        # fig.suptitle('Discrete information event timeline')
        tweets_by_hour.plot.bar(ax=axs[0], x=(tweets_by_hour.index + 24) % 24,
                                y=tweets_by_hour.values, ylabel='n tweets').set_title('Number of tweets/hour:')
        # tweets_by_hour.plot.bar(ax=axs[1], x=(tweets_by_hour.index + 24) % 24,
        #                        y=elontweets['LikeCount'].groupby(elontweets['Datetime'].dt.hour).sum(),
        #                        ylabel='n likes').set_title('Number of likes/hour:')
        tweets_by_hour.plot.bar(ax=axs[1], x=(tweets_by_hour.index + 24) % 24,
                                y=elontweets['LikeCount'].groupby(
                                    elontweets['Datetime'].dt.hour).sum() / tweets_by_hour.values, ylabel='n likes',
                                xlabel='hour').set_title('Normalised number of likes/tweet/hour:')
        tweets_by_hour.plot.bar(ax=axs[2], x=(tweets_by_hour.index + 24) % 24,
                                y=elontweets['RetweetCount'].groupby(
                                    elontweets['Datetime'].dt.hour).sum() / tweets_by_hour.values, ylabel='n likes',
                                xlabel='hour').set_title('Normalised number of retweets/tweet/hour:')

        plt.tight_layout()
        plt.draw()

        print("number of tweets/year")
        print(elontweets['Datetime'].dt.year.value_counts().sort_index())
        print("number of Likes/tweet/year")
        print(elontweets['LikeCount'].groupby(elontweets['Datetime'].dt.year).sum() / tweets_by_year.values)
        print("number of Retweets/tweet/year")
        print(elontweets['RetweetCount'].groupby(elontweets['Datetime'].dt.year).sum() / tweets_by_year.values)

    # plot RH data for visualisation
    if plot_RH_data:
        RH_Data = pd.read_csv('Data/RHdata/popularity_export/SBUX.csv', parse_dates=['timestamp'])
        RH_Data['timestamp'] = pd.to_datetime(RH_Data['timestamp']) - pd.Timedelta(hours=4)
        RH_Data['Datetime'] = RH_Data['timestamp']
        RH_Data['Datetime'] = pd.to_datetime(RH_Data['Datetime'], utc=True)
        RH_Data = RH_Data.sort_values(by=['Datetime'], ascending=True)

        fig = plt.figure()
        fig.suptitle('Robinhood accounts $TSLA ownership changes')
        RH_Data.plot(x='Datetime', y='users_holding', xlabel='time', ylabel='users holding').set(xlim=(
            [datetime.datetime.strptime('07/08/18', '%d/%m/%y'), datetime.datetime.strptime('08/08/18', '%d/%m/%y')]),
            ylim=([75400, 77000]))
        plt.xticks(rotation=45)
        plt.axvline(x=datetime.datetime.strptime('2018-08-07 12:48:13', '%Y-%m-%d %H:%M:%S'), color="green",
                    label="tweet Elon Musk")
        plt.legend()
        plt.draw()

plt.show()

print(vif_tables.to_latex())
