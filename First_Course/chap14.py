# pandas one

# 14.1
import datetime as dt
import yfinance as yf
import pandas as pd

ticker_list = {'INTC': 'Intel',
               'MSFT': 'Microsoft',
               'IBM': 'IBM',
               'BHP': 'BHP',
               'TM': 'Toyota',
               'AAPL': 'Apple',
               'AMZN': 'Amazon',
               'C': 'Citigroup',
               'QCOM': 'Qualcomm',
               'KO': 'Coca-Cola',
               'GOOG': 'Google'}

def read_data(ticker_list,
          start=dt.datetime(2021, 1, 1),
          end=dt.datetime(2021, 12, 31)):
    """
    This function reads in closing price data from Yahoo
    for each tick in the ticker_list.
    """
    ticker = pd.DataFrame()

    for tick in ticker_list:
        stock = yf.Ticker(tick)
        prices = stock.history(start=start, end=end)

        # Change the index to date-only
        prices.index = pd.to_datetime(prices.index.date)
        
        closing_prices = prices['Close']
        ticker[tick] = closing_prices

    return ticker

ticker = read_data(ticker_list)

pct_change_price = 100*(ticker.iloc[-1]/ticker.iloc[0] - 1)
pct_change_price = pct_change_price.sort_values()
pct_change_price.index = pct_change_price.index.map(lambda c:ticker_list[c])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax = pct_change_price.plot(kind="bar")
ax.set_xlabel = "Stock"
ax.set_ylabel = "Percent Growth through 2021"
ax.set_ylim(-10,70)
ax.set_yticks(range(-10,70,5))
plt.show()


# 14.2
indices_list = {'^GSPC': 'S&P 500',
               '^IXIC': 'NASDAQ',
               '^DJI': 'Dow Jones',
               '^N225': 'Nikkei'}

ticker2 = read_data(indices_list, start=dt.datetime(1900,1,1), end=dt.datetime(2021,12,31))

def dt_month(d:dt.datetime):
    return dt.datetime(d.year, d.month,1)

ticker2["yearmonth"] = ticker2.index.map(dt_month)
monthly_groups = ticker2.groupby("yearmonth").mean()

yoy_monthly = monthly_groups.pct_change(periods=11,axis='rows')

fix, axes = plt.subplots(2,2)
i = 0
for (stock, name) in indices_list.items():
    x = yoy_monthly.loc[yoy_monthly[stock].notna()].index
    y = yoy_monthly.loc[yoy_monthly[stock].notna()][stock]
    ax = axes[(i % 2),i//2]
    ax.plot(x,y)
    ax.set_title(name)
    i +=1
plt.show()
