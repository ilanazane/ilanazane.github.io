---
layout: post 
title: "Feature Engineering for Alpha Factor Research" 
date: 2024-02-19
--- 

Alpha factors are mathematical expressions or models that aim to quantify the skill of an investment strategy in generating excess returns beyond what would be expected given the associated risks i.e. it represents the active return of a portfolio. 

Here is the data that we are going to use:

The wiki prices is a NASDAQ dataset that has stock prices, dividends, and splits for 3000 US publicly-traded companies
[wiki prices data](https://data.nasdaq.com/tables/WIKIP/WIKI-PRICES/export)

The US Equities Meta-Data can be found on my github. Beware it is a large file ~1.8 GB. 
[us equities meta data](https://github.com/ilanazane/ML-for-Algorithmic-Trading/tree/main/Feature-Engineering)



## Imports
```python
%pip install pandas_datareader
%pip install statsmodels

```


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pandas_datareader.data as web

import seaborn as sns

from statsmodels.regression.rolling import RollingOLS

import statsmodels.api as sm

from datetime import datetime
```


```python
START = 2000
END = 2018

idx = pd.IndexSlice
```
## Read and Store Data

```python
# parse prices data 

df_prices = (pd.read_csv('WIKI_PRICES.csv',
                 parse_dates = ['date'],
                 index_col = ['date','ticker'],
                 infer_datetime_format=True)).sort_index()

df_screener = pd.read_csv('us_equities_meta_data.csv')
```


```python
# store price and stock data 

with pd.HDFStore('assets.h5') as store:
    store.put('data_prices',df_prices)
    store.put('data_screener',df_screener)
```


```python
# read in prices and stocks 
'''
get all the dates between START and END and 
then get the adj_close column 
reshape df by turning the tickers into the columns and have the adj value for every date 
we convert the rows into columns so that we can easily access the tickers later on 
'''


with pd.HDFStore('assets.h5') as store:
    prices = (store['data_prices']
              .loc[idx[str(START):str(END), :], 'adj_close']
              .unstack('ticker'))
    stocks = store['data_screener'].loc[:, ['ticker','marketcap', 'ipoyear', 'sector']]
```


```python
prices.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 4706 entries, 2000-01-03 to 2018-03-27
    Columns: 3199 entries, A to ZUMZ
    dtypes: float64(3199)
    memory usage: 114.9 MB



```python
stocks.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6834 entries, 0 to 6833
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   ticker     6834 non-null   object 
     1   marketcap  5766 non-null   float64
     2   ipoyear    3038 non-null   float64
     3   sector     5288 non-null   object 
    dtypes: float64(2), object(2)
    memory usage: 267.0+ KB


## Data Cleaning 
remove stock duplicates and reset the index name for later data manipulation 


```python
stocks = stocks[~stocks.index.duplicated()]
stocks = stocks.set_index('ticker')
```

get all of the common tickers with their price information and metadata


```python
shared = prices.columns.intersection(stocks.index)
```


```python
stocks = stocks.loc[shared, :]
stocks.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2412 entries, A to ZUMZ
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   marketcap  2407 non-null   float64
     1   ipoyear    1065 non-null   float64
     2   sector     2372 non-null   object 
    dtypes: float64(2), object(1)
    memory usage: 75.4+ KB



```python
prices = prices.loc[:, shared]
prices.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 4706 entries, 2000-01-03 to 2018-03-27
    Columns: 2412 entries, A to ZUMZ
    dtypes: float64(2412)
    memory usage: 86.6 MB



```python
assert prices.shape[1] == stocks.shape[0]
```

## Create a Monthly Return Series


```python
monthly_prices = prices.resample('M').last()
```

```python
monthly_prices.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 219 entries, 2000-01-31 to 2018-03-31
    Freq: M
    Columns: 2412 entries, A to ZUMZ
    dtypes: float64(2412)
    memory usage: 4.0 MB


Here, we calculate the precentage change of monthly prices over a lag period. This represents the return over the specified number of months.

Clip outliers in the data based on the ```outlier_cutoff``` quantile and the ```1- outlier_cutoff``` to their respective quantile values. Add 1 to the clipped values. This is to handle returns and convert percentage changes back to absolute returns. 

Raise each value to the power of $$\frac{1}{lag}$$. This is used to annualize returns when the lag is representative of months.




```python
outlier_cutoff = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]
for lag in lags:
    data[f'return_{lag}m'] = (monthly_prices
                           .pct_change(lag) 
                           .stack()         
                           .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                  upper=x.quantile(1-outlier_cutoff))) 
                           .add(1)
                           .pow(1/lag)
                           .sub(1)
                           )
data = data.swaplevel().dropna()
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 399525 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 6 columns):
     #   Column      Non-Null Count   Dtype  
    ---  ------      --------------   -----  
     0   return_1m   399525 non-null  float64
     1   return_2m   399525 non-null  float64
     2   return_3m   399525 non-null  float64
     3   return_6m   399525 non-null  float64
     4   return_9m   399525 non-null  float64
     5   return_12m  399525 non-null  float64
    dtypes: float64(6)
    memory usage: 19.9+ MB


## Drop Stocks with Less than 10 Years of Returns 


```python
min_obs = 120

# get the number of observations for each ticker 
nobs = data.groupby(level='ticker').size()

# get the indices of the tickers where the num of observations is greater than min_obs
keep = nobs[nobs>min_obs].index

# get the data with appropriate tickers and their respective data 
data = data.loc[idx[keep,:], :]
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 360752 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 6 columns):
     #   Column      Non-Null Count   Dtype  
    ---  ------      --------------   -----  
     0   return_1m   360752 non-null  float64
     1   return_2m   360752 non-null  float64
     2   return_3m   360752 non-null  float64
     3   return_6m   360752 non-null  float64
     4   return_9m   360752 non-null  float64
     5   return_12m  360752 non-null  float64
    dtypes: float64(6)
    memory usage: 18.0+ MB



```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>return_1m</th>
      <th>return_2m</th>
      <th>return_3m</th>
      <th>return_6m</th>
      <th>return_9m</th>
      <th>return_12m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>360752.000000</td>
      <td>360752.000000</td>
      <td>360752.000000</td>
      <td>360752.000000</td>
      <td>360752.000000</td>
      <td>360752.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.012255</td>
      <td>0.009213</td>
      <td>0.008181</td>
      <td>0.007025</td>
      <td>0.006552</td>
      <td>0.006296</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.114236</td>
      <td>0.081170</td>
      <td>0.066584</td>
      <td>0.048474</td>
      <td>0.039897</td>
      <td>0.034792</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.329564</td>
      <td>-0.255452</td>
      <td>-0.214783</td>
      <td>-0.162063</td>
      <td>-0.131996</td>
      <td>-0.114283</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.046464</td>
      <td>-0.030716</td>
      <td>-0.023961</td>
      <td>-0.014922</td>
      <td>-0.011182</td>
      <td>-0.009064</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.009448</td>
      <td>0.009748</td>
      <td>0.009744</td>
      <td>0.009378</td>
      <td>0.008982</td>
      <td>0.008726</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.066000</td>
      <td>0.049249</td>
      <td>0.042069</td>
      <td>0.031971</td>
      <td>0.027183</td>
      <td>0.024615</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.430943</td>
      <td>0.281819</td>
      <td>0.221789</td>
      <td>0.154555</td>
      <td>0.124718</td>
      <td>0.106371</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.clustermap(data.corr('spearman'), annot=True, center=0, cmap='Blues');
```


    
![image]({{site.url}}/assets/images/feature_engineering_files/feature_engineering_26_0.png)




```python
data.index.get_level_values('ticker').nunique()
```




    1838



## Rolling Factor Betas

In this section we are going to estimate the factor exposure of the listed stocks in the database according to the Fama-French Five-Factor Model.  

```Mkt-Rf``` is the market risk premium $$(R_m - R_f)$$. The difference between these two factors represents the additional return investors demand for bearing the systemic risk associated with the market.  

```SMB``` is small minus big. This factor represents the spread between small-cap and large-cap stocks. 

SMB is typically measured in terms of market capitalization which is represented as $$MarketCapitalization = currentStockPrice * TotalNumberOfOutstandingShares$$

```HML``` is high minus low, which is the spread between high book-to-market and low book-to-market stocks. The book-to-market ratio is $$\frac{shareholder'sEquity}{marketCapitalization}$$.

```RMW``` is robust minus weak. This compares the returns of firms with higher operating profitability and those with weak operating probitability

```CMA``` is conservative minus aggressive and it gauges the difference between companies that invest aggressively and those that do so more conservatively. 



```python
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2000')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 289 entries, 2000-01-31 to 2024-01-31
    Freq: M
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Mkt-RF  289 non-null    float64
     1   SMB     289 non-null    float64
     2   HML     289 non-null    float64
     3   RMW     289 non-null    float64
     4   CMA     289 non-null    float64
    dtypes: float64(5)
    memory usage: 13.5 KB



```python
factor_data = factor_data.join(data['return_1m']).sort_index()
factor_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 360752 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 6 columns):
     #   Column     Non-Null Count   Dtype  
    ---  ------     --------------   -----  
     0   Mkt-RF     360752 non-null  float64
     1   SMB        360752 non-null  float64
     2   HML        360752 non-null  float64
     3   RMW        360752 non-null  float64
     4   CMA        360752 non-null  float64
     5   return_1m  360752 non-null  float64
    dtypes: float64(6)
    memory usage: 18.0+ MB


Unlike traditional linear regression that considers an entire dataset, rolling linear regression calculates regresion coefficients and other statistics for a specified window of consecutive data points and then moves a window forward one observation at a time. 
Rolling Linear Regression is helpful when delaing with non-stationary time series data. 

```python
T = 24
betas = (factor_data.groupby(level='ticker',
                             group_keys=False)
         .apply(lambda x: RollingOLS(endog=x.return_1m,
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(T, x.shape[0]-1))
                .fit(params_only=True)
                .params
                .drop('const', axis=1)))
```


```python
betas.describe().join(betas.sum(1).describe().to_frame('total'))

```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mkt-RF</th>
      <th>SMB</th>
      <th>HML</th>
      <th>RMW</th>
      <th>CMA</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>318478.000000</td>
      <td>318478.000000</td>
      <td>318478.000000</td>
      <td>318478.000000</td>
      <td>318478.000000</td>
      <td>360752.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.979365</td>
      <td>0.626588</td>
      <td>0.122610</td>
      <td>-0.062073</td>
      <td>0.016754</td>
      <td>1.485997</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.918116</td>
      <td>1.254249</td>
      <td>1.603524</td>
      <td>1.908446</td>
      <td>2.158982</td>
      <td>3.306487</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-9.805604</td>
      <td>-10.407516</td>
      <td>-15.382504</td>
      <td>-23.159702</td>
      <td>-18.406854</td>
      <td>-33.499590</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.463725</td>
      <td>-0.118767</td>
      <td>-0.707780</td>
      <td>-0.973586</td>
      <td>-1.071697</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.928902</td>
      <td>0.541623</td>
      <td>0.095292</td>
      <td>0.037585</td>
      <td>0.040641</td>
      <td>1.213499</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.444882</td>
      <td>1.304325</td>
      <td>0.946760</td>
      <td>0.950267</td>
      <td>1.135600</td>
      <td>3.147199</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.855709</td>
      <td>10.297453</td>
      <td>15.038572</td>
      <td>17.079472</td>
      <td>16.671709</td>
      <td>34.259432</td>
    </tr>
  </tbody>
</table>
</div>

```python
cmap = sns.diverging_palette(10,220,as_cmap=True)
sns.clustermap(betas.corr(),annot=True,cmap=cmap,center=0);
```
 
![image]({{site.url}}/assets/images/feature_engineering_files/feature_engineering_33_0.png)    


```python
data = (data.join(betas.groupby(level='ticker').shift()))
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 360752 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 11 columns):
     #   Column      Non-Null Count   Dtype  
    ---  ------      --------------   -----  
     0   return_1m   360752 non-null  float64
     1   return_2m   360752 non-null  float64
     2   return_3m   360752 non-null  float64
     3   return_6m   360752 non-null  float64
     4   return_9m   360752 non-null  float64
     5   return_12m  360752 non-null  float64
     6   Mkt-RF      316640 non-null  float64
     7   SMB         316640 non-null  float64
     8   HML         316640 non-null  float64
     9   RMW         316640 non-null  float64
     10  CMA         316640 non-null  float64
    dtypes: float64(11)
    memory usage: 39.8+ MB


## Impute Mean for Missing Factor Betas


```python
data.loc[:, factors] = data.groupby('ticker')[factors].apply(lambda x: x.fillna(x.mean()))
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 360752 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 11 columns):
     #   Column      Non-Null Count   Dtype  
    ---  ------      --------------   -----  
     0   return_1m   360752 non-null  float64
     1   return_2m   360752 non-null  float64
     2   return_3m   360752 non-null  float64
     3   return_6m   360752 non-null  float64
     4   return_9m   360752 non-null  float64
     5   return_12m  360752 non-null  float64
     6   Mkt-RF      360752 non-null  float64
     7   SMB         360752 non-null  float64
     8   HML         360752 non-null  float64
     9   RMW         360752 non-null  float64
     10  CMA         360752 non-null  float64
    dtypes: float64(11)
    memory usage: 39.8+ MB


## Momentum Factors 

Momentum represents the speed or velocity in which prices change in a publicly traded security. Momentum is caluclated by taking the return of the equal weighted average of the 30% highest performing stocks minus the return of the equal weighted average of the 30% lowest performing stocks.


Here, we create multiple momentum factors based on different lag periods and an additional momentum factor based on the difference between the 12 month and 3 month returns.


```python
for lag in [2,3,6,9,12]:
    # for each value in the loop, calculate new column
    # momentum is computed as the difference between the return for that lag period 
    # and the reutrn for the most recent month 
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)
    
data[f'momentum_3_12'] = data[f'return_12m'].sub(data.return_3m)

```

## Date Indicators 


```python
# date indicators 

dates = data.index.get_level_values('date')
data['year'] = dates.year 
data['month'] = dates.month
```

## Lagged Returns


```python
# to use lagged values as input variables or features associated with the current observations 
# we use the shift function to move historical returns up to the current period 

for t in range(1,7):
    data[f'return_1m_t-{t}'] = data.groupby(level='ticker').return_1m.shift(t)
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 360752 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 25 columns):
     #   Column         Non-Null Count   Dtype  
    ---  ------         --------------   -----  
     0   return_1m      360752 non-null  float64
     1   return_2m      360752 non-null  float64
     2   return_3m      360752 non-null  float64
     3   return_6m      360752 non-null  float64
     4   return_9m      360752 non-null  float64
     5   return_12m     360752 non-null  float64
     6   Mkt-RF         360752 non-null  float64
     7   SMB            360752 non-null  float64
     8   HML            360752 non-null  float64
     9   RMW            360752 non-null  float64
     10  CMA            360752 non-null  float64
     11  momentum_2     360752 non-null  float64
     12  momentum_3     360752 non-null  float64
     13  momentum_6     360752 non-null  float64
     14  momentum_9     360752 non-null  float64
     15  momentum_12    360752 non-null  float64
     16  momentum_3_12  360752 non-null  float64
     17  year           360752 non-null  int64  
     18  month          360752 non-null  int64  
     19  return_1m_t-1  358914 non-null  float64
     20  return_1m_t-2  357076 non-null  float64
     21  return_1m_t-3  355238 non-null  float64
     22  return_1m_t-4  353400 non-null  float64
     23  return_1m_t-5  351562 non-null  float64
     24  return_1m_t-6  349724 non-null  float64
    dtypes: float64(23), int64(2)
    memory usage: 78.3+ MB


## Target: Holding Period Returns

Holding period returns are the total return an investor earns or loses from holding an investment over a specific period of time. $$ HPR = \frac{(endingValue - beginningValue + income)}{beginningValue}*100\%$$

Use the normalized period returns computed previously and shift them back to align them with the current financial features  


```python
for t in [1,2,3,6,12]:
    data[f'target_{t}m'] = data.groupby(level='ticker')[f'return_{t}m'].shift(-t)
```

```python
cols = ['target_1m',
        'target_2m',
        'target_3m', 
        'return_1m',
        'return_2m',
        'return_3m',
        'return_1m_t-1',
        'return_1m_t-2',
        'return_1m_t-3']

data[cols].dropna().sort_index().head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>target_1m</th>
      <th>target_2m</th>
      <th>target_3m</th>
      <th>return_1m</th>
      <th>return_2m</th>
      <th>return_3m</th>
      <th>return_1m_t-1</th>
      <th>return_1m_t-2</th>
      <th>return_1m_t-3</th>
    </tr>
    <tr>
      <th>ticker</th>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">A</th>
      <th>2001-04-30</th>
      <td>-0.140220</td>
      <td>-0.087246</td>
      <td>-0.098192</td>
      <td>0.269444</td>
      <td>0.040966</td>
      <td>-0.105747</td>
      <td>-0.146389</td>
      <td>-0.329564</td>
      <td>-0.003653</td>
    </tr>
    <tr>
      <th>2001-05-31</th>
      <td>-0.031008</td>
      <td>-0.076414</td>
      <td>-0.075527</td>
      <td>-0.140220</td>
      <td>0.044721</td>
      <td>-0.023317</td>
      <td>0.269444</td>
      <td>-0.146389</td>
      <td>-0.329564</td>
    </tr>
    <tr>
      <th>2001-06-30</th>
      <td>-0.119692</td>
      <td>-0.097014</td>
      <td>-0.155847</td>
      <td>-0.031008</td>
      <td>-0.087246</td>
      <td>0.018842</td>
      <td>-0.140220</td>
      <td>0.269444</td>
      <td>-0.146389</td>
    </tr>
    <tr>
      <th>2001-07-31</th>
      <td>-0.073750</td>
      <td>-0.173364</td>
      <td>-0.080114</td>
      <td>-0.119692</td>
      <td>-0.076414</td>
      <td>-0.098192</td>
      <td>-0.031008</td>
      <td>-0.140220</td>
      <td>0.269444</td>
    </tr>
    <tr>
      <th>2001-08-31</th>
      <td>-0.262264</td>
      <td>-0.083279</td>
      <td>0.009593</td>
      <td>-0.073750</td>
      <td>-0.097014</td>
      <td>-0.075527</td>
      <td>-0.119692</td>
      <td>-0.031008</td>
      <td>-0.140220</td>
    </tr>
    <tr>
      <th>2001-09-30</th>
      <td>0.139130</td>
      <td>0.181052</td>
      <td>0.134010</td>
      <td>-0.262264</td>
      <td>-0.173364</td>
      <td>-0.155847</td>
      <td>-0.073750</td>
      <td>-0.119692</td>
      <td>-0.031008</td>
    </tr>
    <tr>
      <th>2001-10-31</th>
      <td>0.224517</td>
      <td>0.131458</td>
      <td>0.108697</td>
      <td>0.139130</td>
      <td>-0.083279</td>
      <td>-0.080114</td>
      <td>-0.262264</td>
      <td>-0.073750</td>
      <td>-0.119692</td>
    </tr>
    <tr>
      <th>2001-11-30</th>
      <td>0.045471</td>
      <td>0.054962</td>
      <td>0.045340</td>
      <td>0.224517</td>
      <td>0.181052</td>
      <td>0.009593</td>
      <td>0.139130</td>
      <td>-0.262264</td>
      <td>-0.073750</td>
    </tr>
    <tr>
      <th>2001-12-31</th>
      <td>0.064539</td>
      <td>0.045275</td>
      <td>0.070347</td>
      <td>0.045471</td>
      <td>0.131458</td>
      <td>0.134010</td>
      <td>0.224517</td>
      <td>0.139130</td>
      <td>-0.262264</td>
    </tr>
    <tr>
      <th>2002-01-31</th>
      <td>0.026359</td>
      <td>0.073264</td>
      <td>-0.003306</td>
      <td>0.064539</td>
      <td>0.054962</td>
      <td>0.108697</td>
      <td>0.045471</td>
      <td>0.224517</td>
      <td>0.139130</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 360752 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 30 columns):
     #   Column         Non-Null Count   Dtype  
    ---  ------         --------------   -----  
     0   return_1m      360752 non-null  float64
     1   return_2m      360752 non-null  float64
     2   return_3m      360752 non-null  float64
     3   return_6m      360752 non-null  float64
     4   return_9m      360752 non-null  float64
     5   return_12m     360752 non-null  float64
     6   Mkt-RF         360752 non-null  float64
     7   SMB            360752 non-null  float64
     8   HML            360752 non-null  float64
     9   RMW            360752 non-null  float64
     10  CMA            360752 non-null  float64
     11  momentum_2     360752 non-null  float64
     12  momentum_3     360752 non-null  float64
     13  momentum_6     360752 non-null  float64
     14  momentum_9     360752 non-null  float64
     15  momentum_12    360752 non-null  float64
     16  momentum_3_12  360752 non-null  float64
     17  year           360752 non-null  int64  
     18  month          360752 non-null  int64  
     19  return_1m_t-1  358914 non-null  float64
     20  return_1m_t-2  357076 non-null  float64
     21  return_1m_t-3  355238 non-null  float64
     22  return_1m_t-4  353400 non-null  float64
     23  return_1m_t-5  351562 non-null  float64
     24  return_1m_t-6  349724 non-null  float64
     25  target_1m      358914 non-null  float64
     26  target_2m      357076 non-null  float64
     27  target_3m      355238 non-null  float64
     28  target_6m      349724 non-null  float64
     29  target_12m     338696 non-null  float64
    dtypes: float64(28), int64(2)
    memory usage: 92.1+ MB


## Create Age Proxy 
Here, we use quintiles of the IPO year as a proxy for company age. This means dividing a set of IPOs into five groups, or quintiles, based on the year in wihch each company went public. Each quintile represents a subset of companies that went public during a specific range of years. 


```python
data = (data
        .join(pd.qcut(stocks.ipoyear, q=5, labels=list(range(1, 6)))
              .astype(float)
              .fillna(0)
              .astype(int)
              .to_frame('age')))
data.age = data.age.fillna(-1)
```

## Create Dynamic Size Proxy


```python
stocks.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2412 entries, A to ZUMZ
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   marketcap  2407 non-null   float64
     1   ipoyear    1065 non-null   float64
     2   sector     2372 non-null   object 
    dtypes: float64(2), object(1)
    memory usage: 139.9+ KB



```python
size_factor = (monthly_prices
               .loc[data.index.get_level_values('date').unique(),
                    data.index.get_level_values('ticker').unique()]
               .sort_index(ascending=False)
               .pct_change()
               .fillna(0)
               .add(1)
               .cumprod())
size_factor.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 207 entries, 2018-03-31 to 2001-01-31
    Columns: 1838 entries, A to ZUMZ
    dtypes: float64(1838)
    memory usage: 2.9 MB


## Create Size Indicator as Declines per Period 


```python
msize = (size_factor
         .mul(stocks
              .loc[size_factor.columns, 'marketcap'])).dropna(axis=1, how='all')
```


```python
data['msize'] = (msize
                 .apply(lambda x: pd.qcut(x, q=10, labels=list(range(1, 11)))
                        .astype(int), axis=1)
                 .stack()
                 .swaplevel())
data.msize = data.msize.fillna(-1)
```

## Combine Data


```python
data = data.join(stocks[['sector']])
data.sector = data.sector.fillna('Unknown')

```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 360752 entries, ('A', Timestamp('2001-01-31 00:00:00', freq='M')) to ('ZUMZ', Timestamp('2018-03-31 00:00:00', freq='M'))
    Data columns (total 33 columns):
     #   Column         Non-Null Count   Dtype  
    ---  ------         --------------   -----  
     0   return_1m      360752 non-null  float64
     1   return_2m      360752 non-null  float64
     2   return_3m      360752 non-null  float64
     3   return_6m      360752 non-null  float64
     4   return_9m      360752 non-null  float64
     5   return_12m     360752 non-null  float64
     6   Mkt-RF         360752 non-null  float64
     7   SMB            360752 non-null  float64
     8   HML            360752 non-null  float64
     9   RMW            360752 non-null  float64
     10  CMA            360752 non-null  float64
     11  momentum_2     360752 non-null  float64
     12  momentum_3     360752 non-null  float64
     13  momentum_6     360752 non-null  float64
     14  momentum_9     360752 non-null  float64
     15  momentum_12    360752 non-null  float64
     16  momentum_3_12  360752 non-null  float64
     17  year           360752 non-null  int64  
     18  month          360752 non-null  int64  
     19  return_1m_t-1  358914 non-null  float64
     20  return_1m_t-2  357076 non-null  float64
     21  return_1m_t-3  355238 non-null  float64
     22  return_1m_t-4  353400 non-null  float64
     23  return_1m_t-5  351562 non-null  float64
     24  return_1m_t-6  349724 non-null  float64
     25  target_1m      358914 non-null  float64
     26  target_2m      357076 non-null  float64
     27  target_3m      355238 non-null  float64
     28  target_6m      349724 non-null  float64
     29  target_12m     338696 non-null  float64
     30  age            360752 non-null  int64  
     31  msize          360752 non-null  float64
     32  sector         360752 non-null  object 
    dtypes: float64(29), int64(3), object(1)
    memory usage: 100.4+ MB


## Store Data 
store the data into an HDF file for later use

```python
with pd.HDFStore('engineered_features.h5') as store:
    store.put('engineered_features', data.sort_index().loc[idx[:, :datetime(2024, 2, 28)], :])
    print(store.info())
```

    <class 'pandas.io.pytables.HDFStore'>
    File path: engineered_features.h5
    /engineered_features            frame        (shape->[360752,33])

