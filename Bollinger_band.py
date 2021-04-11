import pyupbit
import numpy as np
import matplotlib.pyplot as plt
import time

from Upbit_simulator import simulator 

my_wallet = simulator(init_krw=1000000)

df = pyupbit.get_ohlcv('KRW-BTC',count=20000,interval='day')
print(df.keys())

df['ma'] = df['close'].rolling(window=20).mean()
df['std']=df['close'].rolling(window=20).std()
df['low_std']=df['ma']-2*df['std']
df['high_std']=df['ma']+2*df['std']
print(df['low_std'][-1])
print(df)
plt.plot(df['ma'],'k-',label='MA_curve')
plt.plot(df['low_std'],'r--')
plt.plot(df['high_std'],'r--')
plt.legend()
plt.show()

#current_price = pyupbit.get_current_price('KRW-BTC')

def Bollinger_check(ticker):
    df = pyupbit.get_ohlcv(ticker,count=20000,interval='day')
    df['ma'] = df['close'].rolling(window=20).mean()
    df['std'] = df['close'].rolling(window=20).std()
    df['low_std']=df['ma']-1.96*df['std']
    df['high_std']=df['ma']+1.96*df['std']
    
    current_price = pyupbit.get_current_price(ticker)
    if current_price > df['high_std'][-1]:
        #print('High price --> Bot will sell')
        my_wallet.ticker_sell(ticker,current_price)# 2021.04.10 added
    elif current_price < df['low_std'][-1]:
        #print('Low price --> Bot will buy')
        my_wallet.ticker_buy(ticker,current_price)# 2021.04.10 added
    #else:
        #print('Price in the Bollinger band')


import threading

def Periodic_Bollinger_Timer():
    print('3hours passed - Bot will conducting Bollinger band search')
    timer = threading.Timer(3*3600,Periodic_Bollinger_Timer)
    timer.start()
    main_bollinger()
    
def Periodic_balance_show():
    print('1hours passed - Balance?')
    timer2 = threading.Timer(3600,Periodic_balance_show)
    timer2.start()
    my_wallet.Balance_show()

def main_bollinger():
    # Bollinger band check
    tickers = pyupbit.get_tickers(fiat='KRW')
    for ticker in tickers:
        #print(ticker)
        Bollinger_check(ticker)
        time.sleep(0.1)

Periodic_Bollinger_Timer()
Periodic_balance_show()

#while True:
#    for ticker in tickers[:50]:
#        print(ticker)
#        Bollinger_check(ticker)
#        time.sleep(0.1)
