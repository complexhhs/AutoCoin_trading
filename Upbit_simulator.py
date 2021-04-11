import numpy as np
import pandas as pd

class simulator():
    def __init__(self,init_krw=1000000):
        self.krw_balance = init_krw
        self.tickers = {}

    def Balance_write(self,ticker,number,current_price):
        '''
        this method will show ticker information
        info 1. the number of ticker coin
        info 2. the price of 
        '''
        self.tickers[ticker] = {}
        self.tickers[ticker]['coins'] = number
        self.tickers[ticker]['price_per_coin'] = current_price
        self.tickers[ticker]['Total_value'] = self.tickers[ticker]['coins']*self.tickers[ticker]['price_per_coin']

    def Balance_show(self):
        '''
        This method will show every tickers value(KRW) and total price
        '''        
        total_balance = 0
        for t in self.tickers.keys():
            print('Ticker name:', t)
            print('Ticker coins:', self.tickers[t]['coins'])
            print('Ticker price_per_coin:',self.tickers[t]['price_per_coin'])
            print('Ticker value:', self.tickers[t]['Total_value'])
            total_balance += self.tickers[t]['Total_value']
        print('Your total KRW value: ',self.krw_balance)
        print('Your entire coin value(KRW standard)',total_balance)
    
    def ticker_buy(self,ticker,current_price):
        '''
        This method will buy the ticker
        if certain condition is satisfied
        2021.04.10 version --> algorithm can buy 10% of total amount
        '''
        buiable_price = int(self.krw_balance/10)
        if buiable_price <= 1000:
            return None
        num_coins = (buiable_price-0.00005*buiable_price)/current_price
        if ticker not in self.tickers.keys():
            self.tickers[ticker] = {}
        self.tickers[ticker]['coins'] = num_coins
        self.tickers[ticker]['price_per_coin'] = current_price
        self.tickers[ticker]['Total_value'] = self.tickers[ticker]['coins'] * self.tickers[ticker]['price_per_coin']
        self.krw_balance -= (buiable_price + 0.00005*buiable_price)

    def ticker_sell(self,ticker,current_price):
        '''
        This method will sell the ticker 
        if certain condition is satisfied
        2021.04.10 version --> altorithm will sell entire coin
        '''
        if ticker in self.tickers.keys():    
            self.tickers[ticker]['Total_value'] = self.tickers[ticker]['coins']*current_price
            sellable_price = self.tickers[ticker]['Total_value']-0.00005*self.tickers[ticker]['Total_value']
            self.tickers[ticker]['coins'] = 0
            self.tickers[ticker]['price_per_coin'] = current_price
            self.krw_balance += sellable_price