import pandas as pd
import numpy as np 

from stockstats import StockDataFrame as Sdf

csv_file = "AAPL.csv"
moving_avg_start = 15
span = 5

def main():
    df = pd.read_csv(csv_file, header=0)
    df["mv_avg"] = pd.rolling_mean(df["close"], moving_avg_start)
    df["ewma"] = pd.ewma(df["close"], span = moving_avg_start, min_periods = moving_avg_start)

    df["ewma_12"] = pd.ewma(df["close"], span = 12, min_periods = 12)
    df["ewma_26"] = pd.ewma(df["close"], span = 26, min_periods = 26)

    df["macd"] = df["ewma_12"]  - df["ewma_26"] 

    df["k_osc_14"] = osc_n(df["close"])
    stock_df = Sdf.retype(df)
    df["rsi"] = stock_df["rsi_14"]
    df["wr_14"] = stock_df["wr_14"]

    del df['close_-1_s']
    del df['close_-1_d']
    del df['rs_14']
    del df['rsi_14']

    print(df)

def osc_n(c, mv_avg = 14, mv_avg_2=3):
    l, h = pd.rolling_min(c, mv_avg), pd.rolling_max(c, mv_avg)
    k = 100 * (c-l) / (h-1)
    return pd.rolling_mean(k, mv_avg_2)

def calc_rsi(df):
    df.change = df.open - df.close # find out if there is a gain or a loss
    df.gain = df.change [df.change > 0] #column of gain
    df.loss = df.change [df.change < 0]# column of loss
    df.again = df.gain.rolling(center=False,window=10) #find the average gain in the last 10 periods
    df.aloss = df.loss.rolling(center=False,window=10)

if __name__ == "__main__":
    main()
