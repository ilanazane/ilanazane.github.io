# STOCK PRICE PREDICTION AND ANOMOLY DETECTION 

import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# 1. DATA COLLECTION AND PREPARATION 

# fetch historical data 
ticker = yf.Ticker("MSFT")
# print(ticker.info)
data = ticker.history(period = "2y")


# calculate 20 day and 50 day moving averages
data["20_MA"] = data["Close"].rolling(window=20).mean()
data["50_MA"] = data["Close"].rolling(window=50).mean()

print(data.tail()) 

# 2. VISUALIZE STOCK PRICES AND MOVING AVERAGES 
plt.figure(figsize=(14,7))
plt.plot(data["Close"],label = "Close Price")
plt.plot(data["20_MA"],label = "20 day MA", linestyle = "--")
plt.plot(data["50_MA"],label = "50 day MA", linestyle = "--")
plt.title(f"{ticker} Stock Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
# plt.show()

# 3. SIMPLE ANOMOLY DETECTION 

# calculate the rolling standard deviation 
data["20_STD"] = data["Close"].rolling(window=20).std()

# anomolies will be prices that are more than 2 std from the ma
data["Anomoly"] = np.where((data["Close"] > data["20_MA"] + 2 * data["20_STD"])|
                           (data["Close"] < data["20_MA"] - 2 * data["20_STD"]),True,False)

# plot anomolies 
plt.figure(figsize=(14,7))
plt.plot(data["Close"], label='Close Price', color = 'blue')
plt.plot(data["20_MA"],label = "20 day MA", color = "orange")
plt.scatter(data[data["Anomoly"]].index,data[data["Anomoly"]]["Close"],color="red")
plt.title(f"{ticker} Stock Price with Anomolies")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
# plt.show()

# 3. SIMPLE LIQUIDITY SCORING 
# normalize trading volume and rolling volatility 
data["Volume Score"] = (data["Volume"] - data["Volume"].min()) / (data["Volume"].max()- data["Volume"].min())
# higher stability = higher score 
data["Volatility Score"] = 1 - (data["20_STD"] / data["20_STD"].max())

# calculate simple liquidity score 
data["Liquidity Score"] = (data["Volume Score"] + data["Volatility Score"]) / 2

# plot liquidity score 
plt.figure(figsize = (14,7))
plt.plot(data["Liquidity Score"],label = "Liquidity Score", color = "green")
plt.title(f"{ticker} Liquidity Score Over Time")
plt.xlabel("Date")
plt.ylabel("Liquiditty Score (0 to 1")
plt.legend()
plt.show()