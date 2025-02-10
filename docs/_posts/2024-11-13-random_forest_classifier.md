---
layout: post
title: "Random Forests - A Long-Short Strategy for Stocks 🌲 "
date: 2024-11-13
categories: Projects
---

In this project we will use a random forest classifier to generate profitable trading signals for the Nikkei 225. 


### Import Statements 

```python
import graphviz
import numpy as np
import pandas as pd 
import yfinance as yf 
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_curve, roc_auc_score

```

### Download Data

```python
ticker = yf.Ticker('^N225')
data = ticker.history(period = '2y')
display(data.head())
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2022-11-14 00:00:00+09:00</th>
      <td>28277.640625</td>
      <td>28305.039062</td>
      <td>27963.470703</td>
      <td>27963.470703</td>
      <td>85100000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-11-15 00:00:00+09:00</th>
      <td>27940.259766</td>
      <td>28038.630859</td>
      <td>27903.269531</td>
      <td>27990.169922</td>
      <td>71200000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-11-16 00:00:00+09:00</th>
      <td>28020.490234</td>
      <td>28069.250000</td>
      <td>27743.150391</td>
      <td>28028.300781</td>
      <td>73200000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-11-17 00:00:00+09:00</th>
      <td>27952.210938</td>
      <td>28029.619141</td>
      <td>27910.009766</td>
      <td>27930.570312</td>
      <td>58900000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-11-18 00:00:00+09:00</th>
      <td>28009.820312</td>
      <td>28045.439453</td>
      <td>27877.779297</td>
      <td>27899.769531</td>
      <td>64800000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


### Clean Data

```python
# drop rows with missing values 
data = data.dropna()
# check for and drop duplicate dates 
data = data[~data.index.duplicated(keep='first')]
```

The features that we use will be Moving Average Convergence Divergence (MACD), Signal Line, Relative Strength Index (RSI), the Simple Moving Average - 20 days, Simple Moving Average - 50 days Exponential Moving Average- 20 days, and Exponential Moving Average- 50 days. The code below shows how to create these features. 

### Feature Engineering

```python
# window lengths for feature calculation 
short_window = 20  # short term ma window 
long_window = 50   # long term ma window 
vol_window = 20    # volatility window 

# simple moving averages (SMA)
data['sma_20'] = data['Close'].rolling(window=short_window).mean()
data['sma_50'] = data['Close'].rolling(window=long_window).mean()

# exponential moving average (EMA)
data['ema_20'] = data['Close'].ewm(span=short_window,adjust=False).mean()
data['ema_50'] = data['Close'].ewm(span=short_window,adjust=False).mean()
```
The RSI measures the speed and magnitude of a security's recent price changes iin order to detect overvalued or undervalued conditions in the price of that security. Typically an RSI > 70 indicates an overbought condition and an RSI < 30 inidicates an oversold condition. 


<mark style="background-color: lightblue">overbought = trading at a higher price than it's worth and is likely to decline</mark> \
<mark style="background-color: lightblue">oversold = tradinig at a lower price than it's worth and is likely to rally </mark> 

*delta = data['Close'].diff(1)* calculates the day-over-day change in the closing price. For example, if a stock price goes from 100 to 102, the delta for that dat would be 2. This produces a series of price changes for each day. 

```python
#relative strength index (RSI)
delta = data['Close'].diff(1) 
gain = delta.where(delta>0,0)
loss = -delta.where(delta<0,0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
```
Relative Strength (RS) is the ration of the average gain to the average loss. If gains are greater than losses over the period RS will be greater than 1, indicating an upward trend. 

```python
rs = avg_gain/avg_loss

data['rsi'] = 100 - (100 / (1 +rs))

# volatility (rolling standard deviation of returns)
data['volatility'] = data['Close'].pct_change().rolling(window=vol_window).std()
 
# momentum (price difference over the period)
data['momentum'] = data['Close'] - data['Close'].shift(short_window)
```
The MACD is a line that fluctates above and below 0 that indicates when the moving averages are converging, crossing, or diverging.

```python 
# moving average convergence divergence (macd)
data['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['macd'] = data['ema_12'] - data['ema_26']
data['signal_line'] = data['macd'].ewm(span=9, adjust=False).mean()
data['macd_histogram'] = data['macd'] - data['signal_line']
data = data.dropna()
display(data.tail())

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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
      <th>sma_20</th>
      <th>sma_50</th>
      <th>ema_20</th>
      <th>ema_50</th>
      <th>rsi</th>
      <th>volatility</th>
      <th>momentum</th>
      <th>ema_12</th>
      <th>ema_26</th>
      <th>macd</th>
      <th>signal_line</th>
      <th>macd_histogram</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2024-11-07 00:00:00+09:00</th>
      <td>39745.230469</td>
      <td>39884.011719</td>
      <td>39020.218750</td>
      <td>39381.410156</td>
      <td>190000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38901.779102</td>
      <td>38211.528594</td>
      <td>38756.994064</td>
      <td>38756.994064</td>
      <td>54.412138</td>
      <td>0.012271</td>
      <td>443.871094</td>
      <td>38860.938570</td>
      <td>38675.031304</td>
      <td>185.907266</td>
      <td>169.430036</td>
      <td>16.477230</td>
    </tr>
    <tr>
      <th>2024-11-08 00:00:00+09:00</th>
      <td>39783.449219</td>
      <td>39818.410156</td>
      <td>39377.871094</td>
      <td>39500.371094</td>
      <td>159300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38912.899609</td>
      <td>38239.331641</td>
      <td>38827.791877</td>
      <td>38827.791877</td>
      <td>54.822500</td>
      <td>0.012138</td>
      <td>222.410156</td>
      <td>38959.312804</td>
      <td>38736.167585</td>
      <td>223.145220</td>
      <td>180.173073</td>
      <td>42.972147</td>
    </tr>
    <tr>
      <th>2024-11-11 00:00:00+09:00</th>
      <td>39417.210938</td>
      <td>39598.738281</td>
      <td>39315.609375</td>
      <td>39533.320312</td>
      <td>122700000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38920.521094</td>
      <td>38264.225625</td>
      <td>38894.985061</td>
      <td>38894.985061</td>
      <td>55.375531</td>
      <td>0.012127</td>
      <td>152.429688</td>
      <td>39047.621652</td>
      <td>38795.215935</td>
      <td>252.405717</td>
      <td>194.619602</td>
      <td>57.786115</td>
    </tr>
    <tr>
      <th>2024-11-12 00:00:00+09:00</th>
      <td>39642.781250</td>
      <td>39866.718750</td>
      <td>39137.890625</td>
      <td>39376.089844</td>
      <td>163000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38909.035547</td>
      <td>38284.312187</td>
      <td>38940.804564</td>
      <td>38940.804564</td>
      <td>59.646135</td>
      <td>0.012092</td>
      <td>-229.710938</td>
      <td>39098.155220</td>
      <td>38838.243632</td>
      <td>259.911588</td>
      <td>207.677999</td>
      <td>52.233589</td>
    </tr>
    <tr>
      <th>2024-11-13 00:00:00+09:00</th>
      <td>39317.148438</td>
      <td>39377.238281</td>
      <td>38600.261719</td>
      <td>38721.660156</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38849.591016</td>
      <td>38291.494766</td>
      <td>38919.933668</td>
      <td>38919.933668</td>
      <td>55.770084</td>
      <td>0.012470</td>
      <td>-1188.890625</td>
      <td>39040.232902</td>
      <td>38829.607819</td>
      <td>210.625083</td>
      <td>208.267416</td>
      <td>2.357668</td>
    </tr>
  </tbody>
</table>
</div>


### Plot MACD 

The MACD histogram shows the difference between the MACD line and the signal line, highlighting momentum shifts and potential trend reversals. The momemntum shifts occur at points where the two lines crossover. 

<mark style="background-color: lightblue">MACD line above signal = potential buying momentum and entering long position</mark> \
<mark style="background-color: lightblue">MACD line below signal = potential selling momentum and entering short position </mark>

Long position means that you are buying stocks with the intention of profitting from its rising value \
Short position means that you are betting on making money from the stocks falling in value. 


```python
# Set up the figure and axes for subplots
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
# Plot the Closing Price
ax1.plot(data['Close'], label='Close Price', color='blue', alpha=0.7)
ax1.set_title("Stock Price and MACD Indicator")
ax1.set_ylabel("Price")
ax1.legend(loc="upper left")
# Plot the MACD and Signal Line
ax2.plot(data['macd'], label='MACD', color='purple', linewidth=1.5)
ax2.plot(data['signal_line'], label='Signal Line', color='orange', linewidth=1.5)
# Plot the MACD Histogram as a bar plot
ax2.bar(data.index, data['macd_histogram'], label='MACD Histogram', color='grey', alpha=0.3)
# Set labels and title for the MACD plot
ax2.set_ylabel("MACD")
ax2.legend(loc="upper left")
# Display the plot
plt.show()
```

![image]({{site.url}}/assets/images/random_forest_classifier_files/random_forest_classifier_9_0.png)


### Define Long - Short Signals


```python
data['position'] = np.nan
# define long position(1) when macd crosses above signal line 
data.loc[data['macd'] > data['signal_line'], 'position'] = 1 
# define short position(-1) when macd crosses below signal line 
data.loc[data['macd'] < data['signal_line'], 'position'] = -1 
```
Use *ffill()* to carry forward the last signal until a new signal is generated. This means the position will be held until there is a crossover on the signal line

```python
data['position'] = data['position'].ffill()
display(data.tail())
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
      <th>sma_20</th>
      <th>sma_50</th>
      <th>ema_20</th>
      <th>ema_50</th>
      <th>rsi</th>
      <th>volatility</th>
      <th>momentum</th>
      <th>ema_12</th>
      <th>ema_26</th>
      <th>macd</th>
      <th>signal_line</th>
      <th>macd_histogram</th>
      <th>position</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2024-11-07 00:00:00+09:00</th>
      <td>39745.230469</td>
      <td>39884.011719</td>
      <td>39020.218750</td>
      <td>39381.410156</td>
      <td>190000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38901.779102</td>
      <td>38211.528594</td>
      <td>38756.994064</td>
      <td>38756.994064</td>
      <td>54.412138</td>
      <td>0.012271</td>
      <td>443.871094</td>
      <td>38860.938570</td>
      <td>38675.031304</td>
      <td>185.907266</td>
      <td>169.430036</td>
      <td>16.477230</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-08 00:00:00+09:00</th>
      <td>39783.449219</td>
      <td>39818.410156</td>
      <td>39377.871094</td>
      <td>39500.371094</td>
      <td>159300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38912.899609</td>
      <td>38239.331641</td>
      <td>38827.791877</td>
      <td>38827.791877</td>
      <td>54.822500</td>
      <td>0.012138</td>
      <td>222.410156</td>
      <td>38959.312804</td>
      <td>38736.167585</td>
      <td>223.145220</td>
      <td>180.173073</td>
      <td>42.972147</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-11 00:00:00+09:00</th>
      <td>39417.210938</td>
      <td>39598.738281</td>
      <td>39315.609375</td>
      <td>39533.320312</td>
      <td>122700000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38920.521094</td>
      <td>38264.225625</td>
      <td>38894.985061</td>
      <td>38894.985061</td>
      <td>55.375531</td>
      <td>0.012127</td>
      <td>152.429688</td>
      <td>39047.621652</td>
      <td>38795.215935</td>
      <td>252.405717</td>
      <td>194.619602</td>
      <td>57.786115</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-12 00:00:00+09:00</th>
      <td>39642.781250</td>
      <td>39866.718750</td>
      <td>39137.890625</td>
      <td>39376.089844</td>
      <td>163000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38909.035547</td>
      <td>38284.312187</td>
      <td>38940.804564</td>
      <td>38940.804564</td>
      <td>59.646135</td>
      <td>0.012092</td>
      <td>-229.710938</td>
      <td>39098.155220</td>
      <td>38838.243632</td>
      <td>259.911588</td>
      <td>207.677999</td>
      <td>52.233589</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-13 00:00:00+09:00</th>
      <td>39317.148438</td>
      <td>39377.238281</td>
      <td>38600.261719</td>
      <td>38721.660156</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38849.591016</td>
      <td>38291.494766</td>
      <td>38919.933668</td>
      <td>38919.933668</td>
      <td>55.770084</td>
      <td>0.012470</td>
      <td>-1188.890625</td>
      <td>39040.232902</td>
      <td>38829.607819</td>
      <td>210.625083</td>
      <td>208.267416</td>
      <td>2.357668</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


Some notes on random forest classifiers: 

Bagging or Bootstrap Aggregating uses replacement. This means that after selecting a smaple from the dataset to put into the training subset, you put it back into the dataset and it can be chosen again for the same subset or a different subset. Each subset can contain duplicate samples and some samples from the original dataset may not even be included in a subset for training. 

Pasting is the opposite, i.e. without replacement. Each sample in the subset us unique. Every subset of data used for training is therefore completely unique. Pasting works better with larger datasets. 

### Build and Train Random Forest Classifier


```python
# define features and target
features = ['macd','signal_line','rsi','sma_20','ema_20','ema_50']
target = 'position'

# prep features and target var 
x = data[features]
y = data[target]

# drop rows with missiing values 
x = x.dropna()
y = y[x.index] # make sure target var matches the features 

# split the data 
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.2,shuffle=True)
# print(len(x_train),len(x_test))

# initialize random forest classifier 
rf_model = RandomForestClassifier(max_depth = 10, min_samples_leaf=5, n_estimators=100, random_state=42,oob_score=True)
# train the model 
rf_model.fit(x_train,y_train)

# make predictions on the test set 
y_pred = rf_model.predict(x_test)
# evaluate the model 
accuracy = accuracy_score(y_test,y_pred)
print('accuracy: ',accuracy)
print('classification report: ')
print(classification_report(y_test,y_pred))
print('confusion matrix: ')
print(confusion_matrix(y_test,y_pred))
```

    accuracy:  0.9438202247191011
    classification report: 
                  precision    recall  f1-score   support
    
            -1.0       0.96      0.87      0.92        31
             1.0       0.93      0.98      0.96        58
    
        accuracy                           0.94        89
       macro avg       0.95      0.93      0.94        89
    weighted avg       0.94      0.94      0.94        89
    
    confusion matrix: 
    [[27  4]
     [ 1 57]]



```python
# check oob score, becuase random forest classfiier immediately creates this as a validation set 
print(rf_model.oob_score_)
```

    0.884180790960452

Random Forest using oob (out of bag) sampling inherently instead of requiring a validation set to be created.

### Plot ROC and Calculate AUC

```python
# predict probabilities for the positive class 
y_probs = rf_model.predict_proba(x_test)[:,1]
# calculate roc curve 
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# calculate auc 
roc_auc = roc_auc_score(y_test, y_probs)
```


```python
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line for random performance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```

![image]({{site.url}}/assets/images/random_forest_classifier_files/random_forest_classifier_18_0.png)


### Hyperparameter Tuning 
Using default hyperparmeters, I achieved an accuracy score of about 83% which isn't bad, but can be better. Hyperparameter tuning will search a series of parameters to figure out which ones are the best.

The results show that the best parameters are:
 max_depth=10 \
 min_samples_leaf=5 \
 n_estimators=25 

 I went back and replaced the default hyperparameters with the above, but found that when n_estimators = 100, the reuslts were better. 

```python
# set the parameters that we want to search 
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}

grid_search = GridSearchCV(estimator=rf_model,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(x_train, y_train)
```

    Fitting 4 folds for each of 180 candidates, totalling 720 fits
    CPU times: user 2.18 s, sys: 331 ms, total: 2.51 s
    Wall time: 29.9 s


<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4,
             estimator=RandomForestClassifier(max_depth=10, min_samples_leaf=5,
                                              oob_score=True, random_state=42),
             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [2, 3, 5, 10, 20],
                         &#x27;min_samples_leaf&#x27;: [5, 10, 20, 50, 100, 200],
                         &#x27;n_estimators&#x27;: [10, 25, 30, 50, 100, 200]},
             scoring=&#x27;accuracy&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=4,
             estimator=RandomForestClassifier(max_depth=10, min_samples_leaf=5,
                                              oob_score=True, random_state=42),
             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [2, 3, 5, 10, 20],
                         &#x27;min_samples_leaf&#x27;: [5, 10, 20, 50, 100, 200],
                         &#x27;n_estimators&#x27;: [10, 25, 30, 50, 100, 200]},
             scoring=&#x27;accuracy&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: RandomForestClassifier</label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(max_depth=10, min_samples_leaf=5, n_estimators=25,
                       oob_score=True, random_state=42)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(max_depth=10, min_samples_leaf=5, n_estimators=25,
                       oob_score=True, random_state=42)</pre></div> </div></div></div></div></div></div></div></div></div>


### Plot Feature Importance

```python
feature_importance = rf_model.feature_importances_
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# visualize one of the classifiers 
dot_data = export_graphviz(rf_model.estimators_[0], 
                             out_file=None, 
                             feature_names = x.columns, 
                             class_names=['-1','1'],
                             filled=True,
                             rounded=True,
                             special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision tree",format='png',cleanup=True)
graph.view()
```
![image]({{site.url}}/assets/images/random_forest_classifier_files/random_forest_classifier_23_0.png)

So we can see that the relative strength index is the most important feature in the dataset. 


![image]({{site.url}}/assets/images/random_forest_classifier_files/decision tree.png)


### Generate Predictions
Finally, we can generate predictions on the entire data set and compare the predicted positions with the actual positions.

```python
data['predicted_position'] = rf_model.predict(x)
final = data[['position', 'predicted_position']]
display(final)
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
      <th>position</th>
      <th>predicted_position</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-26 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-01-27 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-01-30 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-01-31 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-02-01 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-11-07 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-08 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-11 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-12 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-11-13 00:00:00+09:00</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>443 rows × 2 columns</p>
</div>

We can also see where the actual, calculated decision does not match the predicted position. 

```python
display(data.loc[data['position'] != data['predicted_position']])
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
      <th>sma_20</th>
      <th>sma_50</th>
      <th>ema_20</th>
      <th>...</th>
      <th>rsi</th>
      <th>volatility</th>
      <th>momentum</th>
      <th>ema_12</th>
      <th>ema_26</th>
      <th>macd</th>
      <th>signal_line</th>
      <th>macd_histogram</th>
      <th>position</th>
      <th>predicted_position</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2023-02-17 00:00:00+09:00</th>
      <td>27484.599609</td>
      <td>27608.589844</td>
      <td>27466.609375</td>
      <td>27513.130859</td>
      <td>68800000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27467.344141</td>
      <td>27015.462773</td>
      <td>27385.025950</td>
      <td>...</td>
      <td>52.544581</td>
      <td>0.005926</td>
      <td>959.601562</td>
      <td>27507.998171</td>
      <td>27317.894436</td>
      <td>190.103735</td>
      <td>194.787100</td>
      <td>-4.683365</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-02-20 00:00:00+09:00</th>
      <td>27497.130859</td>
      <td>27531.939453</td>
      <td>27426.480469</td>
      <td>27531.939453</td>
      <td>62500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27498.639160</td>
      <td>27012.373555</td>
      <td>27399.017712</td>
      <td>...</td>
      <td>56.923727</td>
      <td>0.005276</td>
      <td>625.900391</td>
      <td>27511.681445</td>
      <td>27333.749623</td>
      <td>177.931823</td>
      <td>191.416045</td>
      <td>-13.484222</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-03-30 00:00:00+09:00</th>
      <td>27827.890625</td>
      <td>27876.380859</td>
      <td>27630.550781</td>
      <td>27782.929688</td>
      <td>82000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27684.621777</td>
      <td>27494.761758</td>
      <td>27570.491005</td>
      <td>...</td>
      <td>38.133818</td>
      <td>0.011013</td>
      <td>266.400391</td>
      <td>27576.015957</td>
      <td>27556.687718</td>
      <td>19.328239</td>
      <td>6.294103</td>
      <td>13.034136</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2023-04-10 00:00:00+09:00</th>
      <td>27658.519531</td>
      <td>27737.490234</td>
      <td>27597.179688</td>
      <td>27633.660156</td>
      <td>48000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27573.137891</td>
      <td>27619.602656</td>
      <td>27687.528395</td>
      <td>...</td>
      <td>62.826132</td>
      <td>0.010656</td>
      <td>-510.310547</td>
      <td>27711.332896</td>
      <td>27663.756052</td>
      <td>47.576844</td>
      <td>63.484004</td>
      <td>-15.907160</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-04-11 00:00:00+09:00</th>
      <td>27895.900391</td>
      <td>28068.390625</td>
      <td>27854.820312</td>
      <td>27923.369141</td>
      <td>64800000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27577.658301</td>
      <td>27630.418828</td>
      <td>27709.989418</td>
      <td>...</td>
      <td>59.318753</td>
      <td>0.010660</td>
      <td>90.408203</td>
      <td>27743.953857</td>
      <td>27682.986651</td>
      <td>60.967206</td>
      <td>62.980644</td>
      <td>-2.013439</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-09-21 00:00:00+09:00</th>
      <td>32865.558594</td>
      <td>32939.890625</td>
      <td>32550.650391</td>
      <td>32571.029297</td>
      <td>107900000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>32713.799609</td>
      <td>32494.285820</td>
      <td>32757.701448</td>
      <td>...</td>
      <td>49.321868</td>
      <td>0.010014</td>
      <td>560.769531</td>
      <td>32873.815310</td>
      <td>32694.564731</td>
      <td>179.250579</td>
      <td>169.147336</td>
      <td>10.103243</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2023-10-16 00:00:00+09:00</th>
      <td>31983.039062</td>
      <td>31999.789062</td>
      <td>31564.310547</td>
      <td>31659.029297</td>
      <td>84500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>32080.802930</td>
      <td>32235.168242</td>
      <td>31981.839379</td>
      <td>...</td>
      <td>40.252909</td>
      <td>0.013162</td>
      <td>-1509.072266</td>
      <td>31870.004290</td>
      <td>32053.219943</td>
      <td>-183.215653</td>
      <td>-221.515003</td>
      <td>38.299350</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2023-10-25 00:00:00+09:00</th>
      <td>31302.509766</td>
      <td>31466.919922</td>
      <td>31195.580078</td>
      <td>31269.919922</td>
      <td>78700000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31597.691309</td>
      <td>32122.819609</td>
      <td>31678.628901</td>
      <td>...</td>
      <td>57.970678</td>
      <td>0.013278</td>
      <td>-1045.130859</td>
      <td>31506.207344</td>
      <td>31778.585800</td>
      <td>-272.378456</td>
      <td>-225.425469</td>
      <td>-46.952987</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-11-01 00:00:00+09:00</th>
      <td>31311.220703</td>
      <td>31601.650391</td>
      <td>31301.509766</td>
      <td>31601.650391</td>
      <td>130100000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31380.244824</td>
      <td>32044.875977</td>
      <td>31406.949473</td>
      <td>...</td>
      <td>40.677205</td>
      <td>0.014841</td>
      <td>363.710938</td>
      <td>31228.198012</td>
      <td>31523.362572</td>
      <td>-295.164560</td>
      <td>-295.774575</td>
      <td>0.610016</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2023-12-04 00:00:00+09:00</th>
      <td>33318.070312</td>
      <td>33324.378906</td>
      <td>33023.039062</td>
      <td>33231.269531</td>
      <td>87300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>33115.909180</td>
      <td>32218.798789</td>
      <td>33032.855349</td>
      <td>...</td>
      <td>63.357467</td>
      <td>0.009593</td>
      <td>1281.378906</td>
      <td>33265.644680</td>
      <td>32885.774493</td>
      <td>379.870187</td>
      <td>404.636944</td>
      <td>-24.766757</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2023-12-26 00:00:00+09:00</th>
      <td>33295.679688</td>
      <td>33312.261719</td>
      <td>33181.359375</td>
      <td>33305.851562</td>
      <td>68300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>33080.057227</td>
      <td>32578.252500</td>
      <td>33076.413488</td>
      <td>...</td>
      <td>48.296399</td>
      <td>0.011044</td>
      <td>-102.539062</td>
      <td>33145.258185</td>
      <td>33015.601761</td>
      <td>129.656424</td>
      <td>125.346315</td>
      <td>4.310109</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2024-01-30 00:00:00+09:00</th>
      <td>36196.640625</td>
      <td>36249.031250</td>
      <td>36039.308594</td>
      <td>36065.859375</td>
      <td>87900000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35215.043555</td>
      <td>34015.763125</td>
      <td>35337.236029</td>
      <td>...</td>
      <td>69.772308</td>
      <td>0.009780</td>
      <td>2384.621094</td>
      <td>35774.784950</td>
      <td>35050.490528</td>
      <td>724.294422</td>
      <td>730.132578</td>
      <td>-5.838156</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-03-08 00:00:00+09:00</th>
      <td>39809.558594</td>
      <td>39989.328125</td>
      <td>39551.601562</td>
      <td>39688.941406</td>
      <td>143300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38830.578320</td>
      <td>36628.270859</td>
      <td>38809.261927</td>
      <td>...</td>
      <td>71.310158</td>
      <td>0.010562</td>
      <td>3569.019531</td>
      <td>39377.822010</td>
      <td>38399.300401</td>
      <td>978.521609</td>
      <td>1009.610745</td>
      <td>-31.089136</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-03-25 00:00:00+09:00</th>
      <td>40798.960938</td>
      <td>40837.179688</td>
      <td>40414.121094</td>
      <td>40414.121094</td>
      <td>101500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39601.740430</td>
      <td>37836.404219</td>
      <td>39429.966488</td>
      <td>...</td>
      <td>53.418080</td>
      <td>0.011011</td>
      <td>1315.441406</td>
      <td>39827.164290</td>
      <td>39119.040156</td>
      <td>708.124134</td>
      <td>686.521700</td>
      <td>21.602435</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2024-03-26 00:00:00+09:00</th>
      <td>40345.039062</td>
      <td>40529.531250</td>
      <td>40280.851562</td>
      <td>40398.031250</td>
      <td>101400000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39659.956445</td>
      <td>37943.367656</td>
      <td>39522.163132</td>
      <td>...</td>
      <td>53.364377</td>
      <td>0.011013</td>
      <td>1164.320312</td>
      <td>39914.989976</td>
      <td>39213.780237</td>
      <td>701.209739</td>
      <td>689.459308</td>
      <td>11.750432</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2024-03-27 00:00:00+09:00</th>
      <td>40517.171875</td>
      <td>40979.359375</td>
      <td>40452.210938</td>
      <td>40762.730469</td>
      <td>121300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39736.116992</td>
      <td>38047.080078</td>
      <td>39640.312402</td>
      <td>...</td>
      <td>56.967091</td>
      <td>0.011133</td>
      <td>1523.210938</td>
      <td>40045.411590</td>
      <td>39328.517291</td>
      <td>716.894299</td>
      <td>694.946306</td>
      <td>21.947993</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2024-05-30 00:00:00+09:00</th>
      <td>38112.769531</td>
      <td>38138.031250</td>
      <td>37617.000000</td>
      <td>38054.128906</td>
      <td>117300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38561.450195</td>
      <td>38971.007187</td>
      <td>38604.989942</td>
      <td>...</td>
      <td>47.636190</td>
      <td>0.008827</td>
      <td>-351.531250</td>
      <td>38606.600319</td>
      <td>38616.011877</td>
      <td>-9.411558</td>
      <td>1.325213</td>
      <td>-10.736770</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-05-31 00:00:00+09:00</th>
      <td>38173.218750</td>
      <td>38526.929688</td>
      <td>38087.609375</td>
      <td>38487.898438</td>
      <td>211000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38572.142578</td>
      <td>38945.956328</td>
      <td>38593.838370</td>
      <td>...</td>
      <td>53.774913</td>
      <td>0.009177</td>
      <td>213.847656</td>
      <td>38588.338491</td>
      <td>38606.521993</td>
      <td>-18.183501</td>
      <td>-2.576530</td>
      <td>-15.606971</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2024-10-18 00:00:00+09:00</th>
      <td>39092.468750</td>
      <td>39186.640625</td>
      <td>38893.519531</td>
      <td>38981.750000</td>
      <td>95700000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38726.598047</td>
      <td>37697.068047</td>
      <td>38707.127747</td>
      <td>...</td>
      <td>44.308282</td>
      <td>0.018217</td>
      <td>2601.578125</td>
      <td>38996.177299</td>
      <td>38542.783793</td>
      <td>453.393506</td>
      <td>444.418413</td>
      <td>8.975093</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2024-11-06 00:00:00+09:00</th>
      <td>38677.949219</td>
      <td>39664.531250</td>
      <td>38662.171875</td>
      <td>39480.671875</td>
      <td>170600000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38879.585547</td>
      <td>38191.185781</td>
      <td>38691.266054</td>
      <td>...</td>
      <td>52.731370</td>
      <td>0.012487</td>
      <td>147.933594</td>
      <td>38766.307372</td>
      <td>38618.520996</td>
      <td>147.786377</td>
      <td>165.310729</td>
      <td>-17.524352</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 21 columns</p>
</div>

