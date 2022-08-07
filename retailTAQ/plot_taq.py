import pandas as pd
import matplotlib.pyplot as plt
import datetime
# some pandas options, I use it to print the tables easier to terminal for inspection
pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 1000)  # or 1000
pd.set_option('display.max_colwidth', 199)  # or 199

# import our csv file with our merged and identified trades
merged_trades = pd.read_csv('/Users/frederic/Documents/Rotterdam School of Management/Thesis /RSM_Thesis_Code/Data/TAQdata/data.csv')

# convert timestamp to datetime and create new column for date and time respectively
merged_trades['timestamp'] = pd.to_datetime(merged_trades['timestamp'], unit='ns')
merged_trades['time'] = merged_trades['timestamp'].dt.time
merged_trades['date'] = merged_trades['timestamp'].dt.date

# create a new dataset only representing retail trades
D_merged_trades = merged_trades[merged_trades['ex'] == 'D']
retail_merged_trades_A = D_merged_trades[D_merged_trades['symbol'] == 'TSLA']
retail_merged_trades = retail_merged_trades_A[(retail_merged_trades_A['BuySellBJZ'] == 1)
                                              | (retail_merged_trades_A['BuySellBJZ'] == -1)]

# create a new dataset only representing non-retail trades
P_merged_trades = merged_trades[merged_trades['ex'] != 'D']
no_retail_merged_trades_A = P_merged_trades[P_merged_trades['symbol'] == 'TSLA']
no_retail_merged_trades = no_retail_merged_trades_A[(no_retail_merged_trades_A['BuySellLRnotBJZ'] == 1)
                                              | (no_retail_merged_trades_A['BuySellLRnotBJZ'] == -1)]


# lets create a column showing sell/buy volume of retail traders
retail_merged_trades['volsellbuy'] = (retail_merged_trades['size']*retail_merged_trades['BuySellBJZ']).cumsum()
no_retail_merged_trades['volsellbuy'] = (no_retail_merged_trades['size']*no_retail_merged_trades['BuySellLRnotBJZ']).cumsum()

fig = plt.figure()
# fig.suptitle('Robinhood accounts $TSLA ownership changes')
retail_merged_trades.plot( x='timestamp', y='volsellbuy', xlabel='time', ylabel='volume', label='Cumulative retail trader volume').set(xlim=([datetime.datetime.strptime('07/08/18', '%d/%m/%y'), datetime.datetime.strptime('08/08/18', '%d/%m/%y')]),ylim=([0,1000000]))
plt.xticks(rotation=45)
plt.axvline(x = datetime.datetime.strptime('2018-08-07 12:48:13', '%Y-%m-%d %H:%M:%S'), color = "green", label = "tweet Elon Musk")
plt.legend()
plt.show()

# lets plot the retail trades

# the current close price of the asset in scatter and line
#plt.scatter(retail_merged_trades['timestamp'],retail_merged_trades['price'], color='k', alpha=1)
plt.figure(figsize=(8, 4))
plt.plot(retail_merged_trades['timestamp'],retail_merged_trades['price'], color='k', alpha=1)

plt.axvline(x=datetime.datetime(2018, 8, 7, 12, 47,10), label='Elon musk tweets taking $TSLA private')
plt.legend()
plt.xlim(datetime.datetime(2018, 8, 7, 12, 30,10), datetime.datetime(2018, 8, 7, 13, 00,10))
plt.xlabel("Date and time")
plt.ylabel("Price of $TSLA")


plt.figure(figsize=(8, 4))
# the normalised volume of buy/sell orders of the retail traders
#plt.scatter(retail_merged_trades['timestamp'],retail_merged_trades['volsellbuy']/1000+320, color='r', alpha=1)
plt.plot(retail_merged_trades['timestamp'],retail_merged_trades['volsellbuy'], color='r', alpha=1)

plt.axvline(x=datetime.datetime(2018, 8, 7, 12, 47,10), label='Elon musk tweets taking $TSLA private')
plt.legend()
plt.xlim(datetime.datetime(2018, 8, 7, 12, 30,10), datetime.datetime(2018, 8, 7, 13, 00,10))
plt.xlabel("Date and time")
plt.ylabel("Cumulative retail trader volume")

plt.figure(figsize=(8, 4))
plt.plot(no_retail_merged_trades['timestamp'],no_retail_merged_trades['volsellbuy'], color='b', alpha=1)

plt.axvline(x=datetime.datetime(2018, 8, 7, 12, 47,10), label='Elon musk tweets taking $TSLA private')
plt.legend()
plt.xlim(datetime.datetime(2018, 8, 7, 12, 30,10), datetime.datetime(2018, 8, 7, 13, 00,10))
plt.xlabel("Date and time")
plt.ylabel("Cumulative overall trading volume")


plt.show()
#plt.plot(df.Close, color='k', alpha=0.7)

#plt.plot(df['bb_bbl'], color='g', alpha=0.7)

#plt.scatter(Buying_dates, Buying_prices, marker='^', color='g', s=500)
