# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 22:59:26 2017

File name : rule_based.py

@author: achauhan39
Version -Final( Apr-09)


"""


import pandas as pd
import numpy as np
import datetime as DT
import matplotlib.pyplot as plt
from itertools import tee, izip

from util import get_data, plot_data
from marketsim import compute_portvals,get_portfolio_stats
from Indicators import get_features,plot_ind
from ML_based import plot_port_bm,plot_Q6,init


PLOT        =  0
START_CASH  = 100000            # starting cash
S_BLOCK     = 200               # allowed block of shares
DAYS_TO_WAIT = 21

def author():
    return 'achauhan39'


    
##########################################################
# build_orders : Newer version -re-enter same position
# if there's forced exit due to 21 day restriciton
#
########################################################### 
def build_orders_v2(df,lookback) :
    
    print "\n\n Running Manual Trader - VERSION-2  \n\n"
    
    orders=[]
    df['prev_psma'] = df['psma'].shift()
    holdings=0
    wait_period = DAYS_TO_WAIT
          
    dates = df.index
    prev_pos = 'NONE'
    
    for dt in dates[lookback-1:] :
        #print dt , df.ix[dt,'AAPL'],df.ix[dt,'psma'] ,df.ix[dt,'bbp'],df.ix[dt,'d_slow'], df.ix[dt,'roc']        
        wait_period -= 1

        # stock Oversold
        if (df.ix[dt,'psma'] < 0.95) and (df.ix[dt,'bbp'] < 0) and (df.ix[dt,'d_slow'] < 25) and (df.ix[dt,'roc']< -15) :
            if (prev_pos == 'NONE') or (prev_pos == 'SHORT' and wait_period<=0):
                print "stock oversold." ,  df.ix[dt,'roc'] , holdings
                df.ix[dt,'signal'] = 'BUY'
                if holdings < S_BLOCK :
                    holdings += S_BLOCK
                    df.ix[dt,'action'] = 'BUY'
                    orders.append([dt,'AAPL','BUY', S_BLOCK])
                    wait_period = DAYS_TO_WAIT
                    prev_pos ='LONG'
            #else:
               #print "Ignoring BUY Signal - ", dt
       
        #stock Overbought       
        elif (df.ix[dt,'psma'] > 1.05) and (df.ix[dt,'bbp'] > 1) and (df.ix[dt,'d_slow'] > 80):
        #and (df.ix[dt,'roc']> 15 or (df.ix[dt,'rel_p'] > 1.95) ):
            if(prev_pos == 'NONE') or (prev_pos == 'LONG' and wait_period<=0) : 
                print "stock Overbought." ,  df.ix[dt,'roc'] ,holdings 
                df.ix[dt,'signal'] = 'SELL'
               
                if holdings > -S_BLOCK :
                    holdings -= S_BLOCK
                    df.ix[dt,'action'] = 'SELL'
                    orders.append([dt,'AAPL','SELL', S_BLOCK])
                    wait_period = DAYS_TO_WAIT
                    prev_pos = 'SHORT'
            #else: 
                    #print "Ignoring SELL Signal - ", dt
           
           


        # Forced exit - re-enter the postion - MUST EXIT at day-21
#        if(wait_period == 0):
#                if(holdings>0):     #long
#                    if (df.ix[dt,'psma'] >= 1) and (df.ix[dt, 'prev_psma'] < 1) :
#                        orders.append([dt,'AAPL','SELL', S_BLOCK])
#                        df.ix[dt,'action'] = 'CLOSE'
#                        holdings=0
#                        wait_period = 0
#                        prev_pos='NONE'
#                    else :
#                        orders.append([dt,'AAPL','SELL', S_BLOCK])
#                        orders.append([dt,'AAPL','BUY', S_BLOCK])
#                        df.ix[dt,'action'] = 'BUY'
#                        wait_period = DAYS_TO_WAIT
#                        prev_pos='LONG'
#                    
#                if(holdings<0):     #short 
#                    if (df.ix[dt,'psma'] <= 1) and (df.ix[dt,'prev_psma'] > 1) :
#                        orders.append([dt,'AAPL','BUY', S_BLOCK])
#                        df.ix[dt,'action'] = 'CLOSE'
#                        prev_pos='NONE'
#                        holdings=0
#                        wait_period = 0
#                    else:
#                        orders.append([dt,'AAPL','BUY', S_BLOCK])
#                        orders.append([dt,'AAPL','SELL', S_BLOCK])
#                        df.ix[dt,'action'] = 'SELL'
#                        wait_period = DAYS_TO_WAIT
#                        prev_pos='SHORT'
            
        # Can't close if wait_period hasn't expired.
        if(wait_period == 0) :
            if(holdings > 0):
                orders.append([dt,'AAPL','SELL', S_BLOCK])
                orders.append([dt,'AAPL','BUY', S_BLOCK])
                df.ix[dt,'action'] = 'SELL'
            else:
                orders.append([dt,'AAPL','BUY', S_BLOCK])
                orders.append([dt,'AAPL','SELL', S_BLOCK])
                df.ix[dt,'action'] = 'BUY'
            wait_period = DAYS_TO_WAIT
    
    # this change is required to have multiple entries on same date
    df_order = pd.DataFrame(orders, columns=['Date','Symbol', 'Order', 'Shares'])
    df_order = df_order.set_index('Date')
    return df_order
 
##########################################################
# Best Strategy
#
##########################################################
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)
    
def best_strategy(sd = DT.datetime(2008,01,01),
    ed = DT.datetime(2009,12,31),
    symbols = ['AAPL'],
    lookback = 14) :
    
    init()
    df = get_features(sd,ed,symbols)
    bm = compute_portvals( "bm_order_in.csv" , start_val = START_CASH)
 
    orders=[]
    
    for(i1,curr), (i2,nxt) in pairwise(df.iterrows()):
        if(nxt['AAPL'] > curr['AAPL']):
                orders.append([i1,'AAPL','BUY', S_BLOCK ])
                orders.append([i2,'AAPL','SELL', S_BLOCK ])
                #print "BUY :",  i1, i2,  curr['AAPL'],nxt['AAPL'] 
                
        if(nxt['AAPL'] < curr['AAPL']) :
                orders.append([i1,'AAPL','SELL', S_BLOCK ])
                orders.append([i2,'AAPL','BUY', S_BLOCK ])
                #print "SELL :",  i1, i2,  curr['AAPL'],nxt['AAPL'] 
        
        
    df_order = pd.DataFrame(orders, columns=['Date','Symbol', 'Order', 'Shares'])
    df_order = df_order.set_index('Date')
    
    df_order.to_csv("best_strategy_orders.csv",index_label='Date')
 
    port_df = compute_portvals("best_strategy_orders.csv", start_val = START_CASH) 
    
    tmp_df = pd.DataFrame(index = bm.index)
    tmp_df['BM'] = bm
    tmp_df['portfolio'] = port_df
    
    tmp_df = tmp_df.fillna(method='bfill')
    tmp_df = tmp_df.fillna(method='ffill')
        
    #tmp_df_norm = tmp_df
    tmp_df_norm = tmp_df/tmp_df.ix[0]
    
    #print tmp_df_norm.head(10)
    
    ax = tmp_df_norm['BM'].plot(label='benchmark',color='k')
    tmp_df_norm['portfolio'].plot( ax=ax, label='portfolio',color='b')
    plt.title("Best Strategy", fontweight="bold" ,color='g')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best')
    plt.ylabel("Normalised Price")
    plt.show()
    
    cr,adr,sddr,sr               = get_portfolio_stats(tmp_df['portfolio'], verbose=True, title="Portfolio")    
    cr_bm, adr_bm,sddr_bm ,sr_bm = get_portfolio_stats(tmp_df['BM'], verbose=True, title="Benchmark")

    
#    df_best = (abs(df['AAPL'] - df['AAPL'].shift(1)) *S_BLOCK) + START_CASH
#
#    tmp_df = pd.DataFrame(index = bm.index)
#    tmp_df['BM'] = benchmark
#    tmp_df['dr'] = df_best
#    tmp_df['port_val'] = df_best.cumsum()
#            
#    tmp_df_norm = tmp_df
#    #tmp_df_norm = tmp_df/tmp_df.ix[0]
#    
#    ax = tmp_df_norm['BM'].plot(label='benchmark',color='k',title="Best Strategy")
#    tmp_df_norm['port_val'].plot( ax=ax, label='portfolio',color='b')
 
##########################################################
# plot_scatter_manual
#
##########################################################    
def plot_scatter_manual(filename , f1='bbp' ,f2='d_slow'):
   
   filename = "scatter_in_sample_manual.csv"  # todo remove
   f1='bbp'
   f2='d_slow'
   
   # Plot data based on Actual label
   df = pd.read_csv(filename)
   buy  = df[df['signal'] == 'BUY']
   sell = df[df['signal'] == 'SELL']
   nada = df[df['signal'] == 'NADA']
   
   plt.scatter(x=buy[f1],  y=buy[f2]  ,color='g' ,label='LONG')
   plt.scatter(x=sell[f1], y=sell[f2] ,color='r', label='SHORT')
   plt.scatter(x=nada[f1], y=nada[f2] ,color='k' ,label='NADA')
   plt.title("1. SSO vs. BB% - Manual Strategy Signal", fontweight="bold" ,color='b')
   plt.xlabel('BB %', fontweight="bold")
   plt.ylabel(f2 ,fontweight="bold")
   plt.xlim(-1.5,1.5)
   plt.ylim(-1.5,1.5)
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   #plt.xticks(rotation=45)
   plt.show()
   

##########################################################
# build_orders_v1  - Doesn't re-enter same position same day
# 
###########################################################   
def build_orders_v1(df,lookback) :
    
    print "\n\n Running Manual Trader - VERSION-1  \n\n"
    
    
    orders=[]
    df['prev_psma'] = df['psma'].shift()
    holdings=0
    wait_period = DAYS_TO_WAIT

    
    #print df.ix[ :, ['psma','prev_psma']]
        
    dates = df.index
    for dt in dates[lookback-1:] :
        #print dt , df.ix[dt,'AAPL'],df.ix[dt,'psma'] ,df.ix[dt,'bbp'],df.ix[dt,'d_slow'], df.ix[dt,'roc']
        
        wait_period -= 1
        
        # stock Oversold
        #if (df.ix[dt,'psma'] < 0.95) and (df.ix[dt,'bbp'] < 0) and (df.ix[dt,'d_slow'] < 15) and (df.ix[dt,'roc']< -15) :     # **
        if (df.ix[dt,'psma'] < 0.8) and (df.ix[dt,'bbp'] < 0.15) and (df.ix[dt,'d_slow'] < 10) and (df.ix[dt,'roc']< -15) : 
        #if (df.ix[dt,'psma_stdz'] > 0.035) and (wait_period <= 0): 
            #print "stock oversold." ,  df.ix[dt,'roc'] , holdings
            df.ix[dt,'signal'] = 'BUY'
            if holdings < S_BLOCK :
                holdings += S_BLOCK
                df.ix[dt,'action'] = 'BUY'
                orders.append([dt,'AAPL','BUY', S_BLOCK ])
                wait_period = DAYS_TO_WAIT
            
            #print holdings, "\n"
       
        #stock Overbought
        #elif (df.ix[dt,'psma'] > 1.05) and (df.ix[dt,'bbp'] > 1) and (df.ix[dt,'d_slow'] > 80) and \
        #(df.ix[dt,'roc']> 15 or (df.ix[dt,'rel_p'] > 1.9 and (df.ix[dt,'d_slow'] > 94)) ):
        
        #elif (df.ix[dt,'psma'] > 1.05) and (df.ix[dt,'bbp'] > 1) and (df.ix[dt,'d_slow'] > 85):        # **
        elif (df.ix[dt,'psma'] > 1) and (df.ix[dt,'bbp'] > 1) and (df.ix[dt,'d_slow'] > 85):
        #elif (df.ix[dt,'psma_stdz'] < -0.035) and (wait_period <= 0):
        #and (df.ix[dt,'roc']> 15 or (df.ix[dt,'rel_p'] > 1.95) ):      
           #print "stock Overbought." ,  df.ix[dt,'roc'] ,holdings 
           df.ix[dt,'signal'] = 'SELL'  
           if holdings > -S_BLOCK :
                holdings -= S_BLOCK
                df.ix[dt,'action'] = 'SELL'
                orders.append([dt,'AAPL','SELL', S_BLOCK ])
                wait_period = DAYS_TO_WAIT
           #print holdings, "\n"
           
           
        # close position   - MUST EXIT at day-21
        else:
          # Can't close if wait_period hasn't expired.
          if(wait_period <= 0) :
              #  SMA crossed upward- SELL
              #if (df.ix[dt,'psma'] >= 1) and (df.ix[dt, 'prev_psma'] < 1) and (holdings > 0) :
              if (wait_period == 0) or ((df.ix[dt,'psma'] >= 1) and (df.ix[dt, 'prev_psma'] < 1)) :
                  if(holdings > 0):
                      holdings = 0
                      wait_period = 0
                      df.ix[dt,'action'] = 'CLOSE'
                      orders.append([dt,'AAPL','SELL', S_BLOCK])
                      #print dt,"close postion SELL"
              
              #  SMA crossed downward - BUY
              #if (df.ix[dt,'psma'] <= 1) and (df.ix[dt,'prev_psma'] > 1) and (holdings < 0)
              if (wait_period == 0) or ((df.ix[dt,'psma'] <= 1) and (df.ix[dt,'prev_psma'] > 1)) :
                  if(holdings < 0):
                      holdings = 0
                      wait_period = 0
                      df.ix[dt,'action'] = 'CLOSE'
                      orders.append([dt,'AAPL','BUY', S_BLOCK])  
                      #print dt,"close postion BUY"
          
    #print orders
    df_order = pd.DataFrame(orders, columns=['Date','Symbol', 'Order', 'Shares'])
    df_order = df_order.set_index('Date')
 
    return df_order
    
#def run_manual_trader(version=1):
#    sd = DT.datetime(2008,01,01)
#    ed = DT.datetime(2009,12,31)
#    symbols = ['AAPL']
#    lookback = 14
#    init()
#
#    #ed = DT.datetime(2008,05,30)    
#    df = get_features(sd,ed,symbols)
#    
#    df= df.fillna(method ='bfill')
#    df['signal'] = 'NADA'
#    df['action'] = 'NADA'
#
#    if(version==1) :
#        orders_df = build_orders_v1(df,lookback)   
#    else:
#        orders_df = build_orders_v2(df,lookback) 
#        
#    orders_df.to_csv("rule_based_orders.csv",index_label='Date')
#    port_df = compute_portvals("rule_based_orders.csv", start_val = START_CASH)  
#    benchmark = compute_portvals( "bm_order_in.csv" , start_val = START_CASH)
#    plot_port_bm(df,port_df,benchmark ,"Manual Strategy - In Sample" )

##########################################################
#
##########################################################    
def Q6(symbols=['AAPL'],lookback=14):
    sd = DT.datetime(2010,01,01)
    ed = DT.datetime(2011,12,31) 
    df = get_features(sd,ed,symbols)    
    df = df.fillna(method ='bfill')
    df['signal'] = 'NADA'
    df['action'] = 'NADA'

    orders_df = build_orders_v1(df,lookback)    
    orders_df.to_csv("rule_based_orders_out.csv",index_label='Date')  
    port_df_out = compute_portvals("rule_based_orders_out.csv", start_val = START_CASH)  
    plot_Q6(port_df_out)



  
##########################################################
# MAIN
# The in sample/training period is January 1, 2008 to December 31 2009.
# The out of sample/testing period is January 1, 2010 to December 31 2011.
# Starting cash is $100,000.
# Allowable positions are: 200 shares long, 200 shares short, 0 shares.
# holding period =21
##########################################################


def run_manual_trader(version=1,scatter=False, out=False) :
 # print author()
    init()
    sd = DT.datetime(2008,01,01)
    ed = DT.datetime(2009,12,31)
    symbols = ['AAPL']
    lookback = 14

    #ed = DT.datetime(2008,05,30)    
    df = get_features(sd,ed,symbols)
    
    df= df.fillna(method ='bfill')
    df['signal'] = 'NADA'
    df['action'] = 'NADA'
    #df['trade'] = 0
    
    if(PLOT) :
        plot_ind(df)

    if(version==1) :
        orders_df = build_orders_v1(df,lookback)   
    else:
        orders_df = build_orders_v2(df,lookback) 
    
    
    orders_df.to_csv("rule_based_orders.csv",index_label='Date')
    #orders_df.to_csv("rule_based_orders.csv",index=False)

# Execute manual strategy on in-sample data     
    port_df = compute_portvals("rule_based_orders.csv", start_val = START_CASH)  
    #port_df.to_csv("tmp.csv")
    
    benchmark = compute_portvals( "bm_order_in.csv" , start_val = START_CASH)
    #plot the data
    plot_port_bm(df,port_df,benchmark ,"Manual Strategy - In Sample" )
    
    if(scatter == True) :
        #create scatter plot of features
        tmp_df = pd.DataFrame(index=df.index)
        tmp_df['bbp'] = df['bbp']
        tmp_df['d_slow'] = df['d_slow']
        tmp_df['signal'] = df['signal']
        
        #normalise    
        tmp_df['bbp']   = (tmp_df['bbp']-tmp_df['bbp'].mean())/tmp_df['bbp'].std()
        tmp_df['d_slow']   = (tmp_df['d_slow']-tmp_df['d_slow'].mean())/tmp_df['d_slow'].std()
        tmp_df.to_csv("scatter_in_sample_manual.csv")
        plot_scatter_manual("scatter_in_sample_manual.csv" , 'bbp','d_slow')
 
    
    
# Compute Best Strategy
    #best_strategy(df, benchmark)
    
# Execute manual strategy on out of sample data
    if(out == True) :
        sd = DT.datetime(2010,01,01)
        ed = DT.datetime(2011,12,31) 
        df = get_features(sd,ed,symbols)    
        df = df.fillna(method ='bfill')
        df['signal'] = 'NADA'
        df['action'] = 'NADA'
    
        orders_df = build_orders_v1(df,lookback)    
        orders_df.to_csv("rule_based_orders_out.csv",index_label='Date')
        #orders_df.to_csv("rule_based_orders.csv",index=False)
    
        # Plot out of sample data      
        port_df_out = compute_portvals("rule_based_orders_out.csv", start_val = START_CASH)  
        bm_out = compute_portvals( "bm_order_out.csv" , start_val = START_CASH)
        plot_port_bm(df,port_df_out, bm_out ,"Manual Strategy - Out-of-Sample" )
    
    
    
    
    


if __name__ == "__main__":
    
    run_manual_trader(version=1,scatter=True, out=True)
    
    
  
    
    
    
    
#    #buy_dates = orders_df.index[orders_df['Order']=='BUY']    
#    #sell_dates = orders_df.index[orders_df['Order']=='SELL'] 
#        
#    buy_dates = df.index[df['action']=='BUY']    
#    sell_dates = df.index[df['action']=='SELL'] 
#    close_dates = df.index[df['action']=='CLOSE'] 
#    
#    # plot normalised
#    # print benchmark.head()
#    # print port_df.head()
#    
#    tmp_df = pd.DataFrame(index = benchmark.index)
#    tmp_df['BM'] = benchmark
#    tmp_df['portfolio'] = port_df
#    
#    tmp_df = tmp_df.fillna(method='bfill')
#    tmp_df = tmp_df.fillna(method='ffill')
#    
#    #print tmp_df.head(25)
#    
#    #tmp_df_norm = tmp_df
#    tmp_df_norm = tmp_df/tmp_df.ix[0]
#    
#    #print tmp_df_norm.head(25)
#    
#    ax = tmp_df_norm['BM'].plot(label='benchmark',color='k',title="Rule based Strategy")
#    tmp_df_norm['portfolio'].plot( ax=ax, label='portfolio',color='b')
#    
#    ymin, ymax = ax.get_ylim()
#    ax.vlines(x=sell_dates,ymin=ymin,ymax=ymax,color='r' )
#    ax.vlines(x=buy_dates ,ymin=ymin,ymax=ymax,color='g')  #, linestyle='--')
#    ax.vlines(x=close_dates ,ymin=ymin,ymax=ymax,color='k', linestyle=':') 
#  
#    
#    ax.set_ylim([ymin,ymax])
#    plt.show()
#    
#    print df.ix[df['action']!= 'NADA' , ['action','signal']]
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  ### test
test=0
if(test) : 
    sd = DT.datetime(2008,01,01)
    ed = DT.datetime(2008,01,10)    
    df = get_features(sd,ed,['AAPL'])
    indx = df.index
    last_trade = indx[0]
    
    orders=[] 
    for dt in df.index :
        #orders.append([dt.strftime('%m/%d/%Y'),'APPL', 'BUY'])
        orders.append([dt,'APPL', 'BUY' ,200])
        orders.append([dt,'APPL', 'SELL' ,200])
    
    #print orders
    
    order_df = pd.DataFrame(orders, columns=['Date','Symbol', 'Order', 'Shares'])
    order_df=order_df.set_index('Date')
    #order_df.to_csv("try.csv", index=False)
    print order_df
    

      
    t_days_left = 21
    prev_row = pd.Series()
    for i ,row in df.iterrows():
        if prev_row.empty == False:  
            t_days_left -=1
            if(t_days_left <= 0):
                print "Can close",  row['AAPL'] , row['psma'], "|" , i
                t_days_left = 21
     
        prev_row = row
  
  
  
  
  
  
	# plot = port_df.plot(title = "" ,ylabel="Normalised price")    
	# # add benchmark to it
    # buy_dates = orders_df.index[orders_df['Order']=='BUY']    
    # sell_dates = orders_df.index[orders_df['Order']=='SELL'] 
    
    # plot = port_val_df.plot()    
    # ymin, ymax = plot.get_ylim()
    # plot.vlines(x=sell_dates, ymin=ymin, ymax=ymax-1,color='r')
    # plot.vlines(x=buy_dates, ymin=ymin, ymax=ymax-1,color='g')
    
    # #plot normalised
    # price_SPY= get_benchmark()
    # #tmp_df = pd.concat([port_val_df,price_SPY],keys=['Portfolio','benchmark'], axis=1)
    # tmp_df = port_val_df
    # tmp_df['BM'] = price_SPY
    # tmp_df_norm = tmp_df/tmp_df.ix[0]
    # plot = tmp_df_norm.plot()
    # ymin, ymax = plot.get_ylim()
    # plot.vlines(x=sell_dates,ymin=ymin,ymax=ymax,color='r')
    # plot.vlines(x=buy_dates ,ymin=ymin,ymax=ymax,color='g')
	
 

##########################################################
# 
#
###########################################################

   
def test_diff_point():
   df = pd.read_csv("test_diff_points.csv",index_col='Date',parse_dates=True)
 
   buy = df[df['act']=='BUY']
   plt.scatter(x=buy.index, y=buy['bbp'] ,color='g')
   plt.scatter(x=buy.index, y=buy['psma'],color='g')
   
   sell = df[df['act']=='SELL']
   plt.scatter(x=sell.index, y=sell['bbp'] ,color='r')
   plt.scatter(x=sell.index, y=sell['psma'],color='r')    
       
   nada = df[df['act']=='nada'] 
   plt.scatter(x=nada.index, y=nada['bbp'] ,color='k')
   plt.scatter(x=nada.index, y=nada['psma'],color='k')  
   
   plt.title("Features X1,X2 - Manual Strategy")
   plt.ylabel("features")
   plt.xticks(rotation=45)
   plt.show()   
   
   
   
   
  

 




#
#        # Forced exit - re-enter the postion - MUST EXIT at day-21
#        if(wait_period == 0):
#                if(holdings>0):     #long
#                    if (df.ix[dt,'psma'] >= 1) and (df.ix[dt, 'prev_psma'] < 1) :
#                        orders.append([dt,'AAPL','SELL', S_BLOCK])
#                        df.ix[dt,'action'] = 'CLOSE'
#                        holdings=0
#                        wait_period = 0
#                    else :
#                        orders.append([dt,'AAPL','SELL', S_BLOCK])
#                        orders.append([dt,'AAPL','BUY', S_BLOCK])
#                        df.ix[dt,'action'] = 'BUY'
#                        wait_period = DAYS_TO_WAIT
#                    
#                if(holdings<0):     #short 
#                    if (df.ix[dt,'psma'] <= 1) and (df.ix[dt,'prev_psma'] > 1) :
#                        orders.append([dt,'AAPL','BUY', S_BLOCK])
#                        df.ix[dt,'action'] = 'CLOSE'
#                        holdings=0
#                        wait_period = 0
#                    else:
#                        orders.append([dt,'AAPL','BUY', S_BLOCK])
#                        orders.append([dt,'AAPL','SELL', S_BLOCK])
#                        df.ix[dt,'action'] = 'SELL'
#                        wait_period = DAYS_TO_WAIT