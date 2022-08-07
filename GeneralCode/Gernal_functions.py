from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as st
from statsmodels.tsa.stattools import adfuller

def save_df(dataframe, path):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(filepath)
    return


def get_df(path):
    filepath = Path(path)
    dataframe = pd.read_csv(filepath)
    return dataframe


POSITIVE = 1
NEUTRAL = 0
NEGATIVE = -1


def categorize_sentiment(sent):
    if sent < -0.3:
        return NEGATIVE
    elif sent < 0.3:
        return NEUTRAL
    return POSITIVE


# https://towardsdatascience.com/granger-causality-and-vector-auto-regressive-model-for-time-series-forecasting-3226a64889a6
def granger_causality_matrix(X_train, variables, lag, test='ssr_chi2test', verbose=False):
    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in dataset.columns:
        for r in dataset.index:
            try:
                test_result = st.grangercausalitytests(X_train[[r, c]], maxlag=lag, verbose=False)
            except:
                pass
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(lag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            dataset.loc[r, c] = min_p_value
    dataset.columns = [var + '_x' for var in variables]
    dataset.index = [var + '_y' for var in variables]
    return dataset


def difx(s):
    return s.diff().combine_first(s)


def adf_test(timeseries):

    #print ('Results of Dickey-Fuller Test:')

   # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    #for key,value in dftest[4].items():
    #    dfoutput['Critical Value (%s)'%key] = value
    #print(dfoutput)
    #print(dftest[0])
    #print(dftest[1])
    #print(dftest[2])

    #print(dftest[4].get('1%'))
    #print("next")
    dftest = adfuller(timeseries, autolag='AIC')
    while True:
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test statistic', 'p-value', '# Lags    Used', 'Number of Observations Used'])
        if dfoutput['p-value'] > 0.05:
            timeseries = difx(timeseries)
            #print(dfoutput['p-value'])
            #print("differentiate")
            dftest = adfuller(timeseries, autolag='AIC')
        else:
            break

    return timeseries, dftest[2]
