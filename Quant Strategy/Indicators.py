"""
Created on Fri Mar 31 11:43:51 2017

File name : Indicators.py

@author: achauhan39
Version -Final( Apr-09)

"""

import pandas as pd
import numpy as np
import datetime as DT
import os
import matplotlib.pyplot as plt


from util import get_data, plot_data
from marketsim import compute_portvals



    
##########################################################
#  get_features 
#
###########################################################    
def get_features(sd,ed,symbol=['AAPL'],lookback=14 ,norm=False) :
    
#    sd = dt.datetime(2008,01,01)   #todo - remove it
#    ed = dt.datetime(2009,12,31)
#    symbol=['AAPL']
#    norm=False
    
    dates = pd.date_range(sd,ed)
    df = get_data(symbol,dates)
                    
    df = df.dropna(axis=0)
    
    df.fillna(method ='ffill')  
    df.fillna(method ='bfill')
    
    df_orig = df
    
    if(norm):
        df = df/df.ix[0]
        print "Normalizing features"
        
    price = df[symbol]
    
    sma = price.rolling(window=lookback,min_periods=lookback).mean()
    r_std = price.rolling(window=lookback,min_periods=lookback).std()
    bb_upper = sma + 2*r_std
    bb_lower = sma - 2*r_std
    
    df['sma'] = sma
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bbp'] = (price - bb_lower)/(bb_upper - bb_lower)
    df['psma'] = price/sma
    df['rel_p'] = df_orig['AAPL']/df_orig['SPY']
    #df['roc'] = 100 *  ( (df['AAPL'] - df['AAPL'].shift(lookback))/df['AAPL'].shift(lookback))
    df['roc'] = 100 *  ( (df['AAPL']/df['AAPL'].shift(lookback-1)) - 1 )
    df['psma_stdz'] = (df['psma']- df['psma'].mean())/df['psma'].std()
    
    
    #compute Stochastic Oscillator
    #df_sos = stochastic(df.index)  
    
    df_so = pd.DataFrame(index=df.index)
    
    filename = os.path.join(os.path.join("..", "data"), "{}.csv".format(str('AAPL')))
    df_temp = pd.read_csv(filename,index_col='Date', parse_dates=True, na_values=['nan'])
    df_so = df_so.join(df_temp)
    
    ''' 
        Fast stochastic calculation
            %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
            %D = 3-day SMA of %K
    '''
    low_min = df_so['Low'].rolling(window=lookback).min()
    high_max = df_so['High'].rolling(window=lookback).max()
    df_so['low_min'] = low_min
    df_so['high_max'] = high_max
    df_so['k_fast'] = (df_so['Close'] - low_min)/(high_max - low_min) * 100
    df_so['d_fast'] = df_so['k_fast'].rolling(window=3,min_periods=3).mean()
    
    """ 
        Slow stochastic calculation
            %K = %D of fast stochastic
            %D = 3-day SMA of %K
    """
    df_so['k_slow'] = df_so['d_fast']
    df_so['d_slow'] = df_so['k_slow'].rolling(window=3,min_periods=3).mean()
    
    #df_so.to_csv("df_so.csv")
    #df.to_csv("df.csv")
    
    merged_df = df.join(df_so)
    #merged_df.to_csv("merged_df.csv")
    
    return merged_df


#def trade(df, lookback=14, hold_p = 21):
#    dates = df.index 
#    for idx in range(lookback-1, len(dates)):
#        dt = dates[idx]
#        #df.ix[dt,'psma'] and df.ix[dt,'bbp'] and df.ix[dt,'d_slow']
#        print dt, df.ix[dt,'psma'] , df.ix[dt,'bbp'] # , df.ix[dt,'d_slow']
        
        
def plot_ind(df) :
    # Bollinger Bands
    ax = df['AAPL'].plot(title="Bollinger Band - AAPL", label='AAPL')
    df['sma'].plot(label='sma', ax=ax,color='k',linewidth=2 )
    ax.fill_between(x=df.index,y1=df['bb_upper'], y2=df['bb_lower'] , color='y', alpha=0.8 , label='b band')
    #df['bb_upper'].plot(label='bb-upper', ax=ax, ls='--',color='g')
    #df['bb_lower'].plot(label='bb-lower', ax=ax ,ls='--',color='g')
    ax.set_ylabel("Normalised Price")
    ax.legend(loc='lower right')
    plt.show()
    
    ax = df['bbp'].plot(title="1.BollingerBand % - AAPL", label='',color='m')
    ax.axhline(y=1,color='gray', ls='-')
    ax.axhline(y=0,color='gray', ls='-')
    ax.axhspan(0,1, alpha=.2,color='y')
    #ax.axhspan(1,ax.get_ylim()[1], alpha=.1,color='g')
    #ax.axhspan(0,ax.get_ylim()[0], alpha=.1,color='r')
    plt.show()
    
    
    # 2. ROC
    ax = (df['roc']/100.0).plot(label='ROC', title= "2. Rate-of-Change(ROC)- APPL",color='m')
    ax = df['AAPL'].plot(label='price')
    ax.axhline(y=.1,color='gray', ls='--')
    ax.axhline(y=-.1,color='gray', ls='--')
    ax.axhspan(0.1,-0.1, alpha=.25,color='y')
    plt.legend(loc='best')
    plt.show()
    
    
    #3.  Price/SMA
    ax = df['AAPL'].plot(label='AAPL',color='gray')
    df['sma'].plot(label='sma', ax=ax,color='k', linewidth=1)
    #plt.show()
    #ax = df['psma'].plot(title="Price/SMA - AAPL", label='',color='m')
    df['psma'].plot(ax=ax, title="3.Price/SMA - AAPL", label='Price/SMA',color='m')
    ax.axhline(y=1.05,color='gray', ls='--')
    ax.axhline(y=0.95,color='gray', ls='--')
    ax.axhspan(1.05,0.95, alpha=.25,color='y')
    plt.legend(loc='best')
    plt.show()
    
    #4.Stochastic Oscillator
    ax= df['Close'].plot(label='close', title= "High/Low price -AAPL",color='k')
    ax.fill_between(x=df.index,y1=df['High'], y2=df['Low'] , color='y', label='High-Low')
    #df['High'].plot(label='high' ,ax=ax)
    #df['Low'].plot(label='low' ,ax=ax)
    #plt.legend(loc='best')
    #plt.show()
    
    #ax = df['Close'].plot(label='', title= "Fast SO")
    #df['k_fast'].plot(label='k_fast', ax=ax)
    #df['d_fast'].plot(label='d_fast', ax=ax , color='m')
    
    ax = df['k_fast'].plot(label='k_fast', title= "FAST - Stochastic Oscillator",color='gray')
    df['d_fast'].plot(label='d_fast', ax=ax , color='b')
    ax.axhspan(20,80, alpha=.2,color='y')
    plt.legend(loc='best')
    plt.show()
    
    ax = df['k_slow'].plot(label='k_slow', title= "4. SLOW - Stochastic Oscillator",color='gray')
    df['d_slow'].plot(label='d_slow', ax=ax , color='m')
    ax.axhspan(20,80, alpha=.2,color='y')
    plt.legend(loc='best')
    plt.show()
    
    # Relative Price
#    ax= df['AAPL'].plot(label='AAPL', title= "Price- AAPL vs. SPY")
#    df['SPY'].plot(label='SPY' ,ax=ax)
#    plt.legend(loc='best')
#    plt.show()
#    
#    ax= df['rel_p'].plot(title="RelativePrice -AAPL",color='m')
#    plt.show()
    

    

    
##########################################################
#  Plot normalized features
#
###########################################################    
def plot_normalised_Indicators( 
    sd = DT.datetime(2008,01,01),
    ed = DT.datetime(2009,12,31),
    symbols = ['AAPL'],
    lookback = 14) :
    
    df = get_features(sd,ed,symbols,lookback, norm=1)
    #df.to_csv("output_indicator.csv")
    plot_ind(df)


if __name__ == "__main__":
   plot_normalised_Indicators()
    
    
    
    
    
    
    

##########################################################
#  Compute stochastic oscillator
#
###########################################################
#def stochastic(index, symbol='AAPL'):
#    
#    df = pd.DateFrame(index=index)
#    
#    filename = os.path.join(base_dir, "{}.csv".format(str(symbol)))
#    df_temp = pd.read_csv(filename,index_col='Date', parse_dates=True, na_values=['nan'])
#    df = df.join(df_temp)
#    
#        #f = os.path.join(base_dir, "{}.csv".format(str(symbol)))
#    #df = pd.read_csv(f , index_col='Date', parse_dates=True, na_values=['nan'])   