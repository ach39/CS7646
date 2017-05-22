"""MC2-P1: Market simulator.
@author : achauhan39

Important -  
	drop date 2011-06-15
	recompute if leverage > 1.5

version - Feb-28-2017
 @ Mar-3 2:40pm : change handling for Jun-15-2011
 
"""


import pandas as pd
import numpy as np
import datetime as dt
import os
import time

from util import get_data, plot_data

LEV_DBG = 0
DEBUG= 0
N=100


    
def author():
    return 'achauhan39'
    
    
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
        
    orders_df = pd.read_csv(orders_file,index_col='Date',parse_dates=True, 
                            usecols=['Date', 'Symbol', 'Order', 'Shares'],na_values=['nan'])
   
    #orders_df = pd.read_csv(orders_file,index_col='Date',parse_dates=True, na_values=['nan'])
           
   
    orders_df.sort_index(inplace=True)          #sort dates
    
    # drop Jun-15-2011 from orders_df    
    #if'2011-06-15' in orders_df.index : 
        #orders_df.drop(pd.Timestamp('2011-06-15'), inplace=True)
        #print " Dropping Jun-15-2011"
      
    """
        1. Create Price data frame.
    """
    sd = orders_df.index.min()
    ed = orders_df.index.max()
    syms = list(orders_df.Symbol.unique())
    
    if DEBUG : print sd,ed, syms
    if LEV_DBG : print orders_df
    
        
    #dates = pd.bdate_range(sd,ed)                       # using business days only. Check if federal holidays needs to dropped  
    dates = pd.date_range(sd,ed) 
    price_all = get_data(syms, dates)
    price_all.fillna(method ='ffill')  
    price_all.fillna(method ='bfill')
    
    price_df = price_all[syms]                          # remove SPY
        
    price_df['cash'] = np.ones(price_df.shape[0])            # Add Cash column
    
    if DEBUG : print price_df.head(n=N)
    #print price_df['2011-06-5' : '2011-06-20']
    
    """
        2. Create Trade data frame
    """
    trade_df = price_df.copy()
    trade_df.ix[:,:] = 0
        
    #iterate over orders index 
    for date in price_df.index:
        if date != pd.to_datetime('2011-06-15'):
            if date in orders_df.index:
                sub_order = orders_df.ix[date :date]
                for i in range(0,sub_order.shape[0]):
                    sym = sub_order.ix[i,'Symbol']
                    order = sub_order.ix[i,'Order']
                    shares = sub_order.ix[i,'Shares']
    
                    if order == 'SELL':
                        trade_df.ix[date,sym] += shares *(-1)
                        trade_df.ix[date,'cash'] += price_df.ix[date,sym] * shares
                    elif order == 'BUY':
                        trade_df.ix[date,sym] += shares
                        trade_df.ix[date,'cash'] += price_df.ix[date,sym] * shares*(-1)
                    else:
                        print "2.Error - No buy or sell"
                
                #print trade_df.ix[date,'cash'] 
        
    if DEBUG : print "TRADING df \n" , trade_df.head(n=N)
    

    """
        3. Create Holdings data frame
    """

    holdings_df = trade_df.copy()
    holdings_df.ix[:,:] = 0    
    holdings_df.ix[0,'cash'] = start_val
   
    # special handling of 1st row
    holdings_df.ix[0,:] +=  trade_df.ix[0,:]
    
    for i in range(1,holdings_df.shape[0]) :
        holdings_df.ix[i,:] =  holdings_df.ix[i-1,:] +  trade_df.ix[i,:]
    
    if DEBUG : print "HOLDING DF  \n", holdings_df.head(n=N)
    
    
    """
        4. Create Value data frame
    """  
    value_df = holdings_df * price_df
    if DEBUG : print"VALUE DF  \n", value_df.head(n=N)
    
    """
        5. Create port_val df
    """
    port_val_df = pd.DataFrame(index=holdings_df.index)
    port_val_df['val'] = value_df.sum(axis=1)
    if DEBUG : print "PORT_VAL  DF  \n", port_val_df.head(n=N)
    if LEV_DBG :print"PORT VAL DF  \n", port_val_df.tail(5)
    
    """
        6. Check Leverage
    """
    lev_df = pd.DataFrame(index=holdings_df.index)
    lev_df['leverage'] = (value_df.ix[: , :-1].abs().sum(axis=1)) / (value_df.ix[: , :-1].sum(axis=1) + value_df['cash'])
        
    lev_offender = lev_df[lev_df['leverage'] > 1.5]
    l_index = list(lev_offender.index)
    if len(l_index)>0:
        date = l_index[0]
        
        if LEV_DBG : print "Leverage exceeded :  dropped index "  , date  
        #is it last day ??
        if date == ed :
            #print "leverage exceeded on last day" 
            temp = orders_df.ix[date :date]
            for i in range(0,temp.shape[0]):
                temp.ix[i,'Shares'] = 0
        else :
            orders_df.drop(date, inplace=True)
			
        orders_df.to_csv('modified_order.csv', sep=',')       
        return compute_portvals( "modified_order.csv", start_val = 1000000)
        
    return port_val_df



def get_portfolio_stats(port_val, rfr=0.0 , sf=252.0):
    
    '''
     Function : get_portfolio_stats
     input    : portfolio value, risk free rate of return, sampling frequency
     outptut  : cumulative return , avg daily return, 
                std_dev of daily return , sharpe ratio
    '''
    
    cr = (port_val[-1]/port_val[0]) -1      # Cummulative Return

    dr = (port_val/port_val.shift(1)) -1    # Daily Return
    dr.ix[0] = 0
    dr = dr[1:]                             # DR- Remove top row

    adr = dr.mean()                         # Average Daily Return
    sddr = dr.std()                         # STD of Daily Return
    dr_rfr_delta = dr-rfr
    sr = np.sqrt(sf) * (dr_rfr_delta.mean()/sddr)     # Sharpe Ratio

    return cr,adr,sddr,sr


def test_code(of , sv = 1000000 ):

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    
    start_date = portvals.index.min()
    end_date = portvals.index.max()
 
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)
    
    port_val_SPY = get_data(['SPY'], dates = pd.date_range(start_date, end_date))
    #if DEBUG : print port_val_SPY.head()
    
    #cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(port_val_SPY.ix[:,'SPY'])
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(port_val_SPY[port_val_SPY.columns[0]])


    # Compare portfolio against $SPX
#    print "Date Range: {} to {}".format(start_date, end_date)
#    print
#    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
#    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
#    print
#    print "Cumulative Return of Fund: {}".format(cum_ret)
#    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
#    print
#    print "Standard Deviation of Fund: {}".format(std_daily_ret)
#    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
#    print
#    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
#    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
#    print
#    print "Final Portfolio Value: {}".format(portvals[-1])
     
#    print "Date Range: {} to {}".format(start_date, end_date)
#    print "num days :"  , (portvals.shape[0])
#    print "last_day_portval: {}".format(portvals[-1])
#    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
#    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    
    print portvals.shape[0] , portvals[-1] , (sharpe_ratio), avg_daily_ret


if __name__ == "__main__":
    
    print author()
    
#    test_code("./Orders_mc2p1_spr2016/orders-01_arti.csv")          # has Jun-15 and weekends    

#    test_code("./Orders_mc2p1_spr2016/orders-01.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-02.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-03.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-04.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-05.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-06.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-07-modified.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-08-modified.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-09-modified.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-10-modified.csv")       
# 
#    test_code("./Orders_mc2p1_spr2016/orders-11-modified.csv")  # failing expected val : 1086640
#    test_code("./Orders_mc2p1_spr2016/orders-12-modified.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-leverage-1.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-leverage-2.csv")
#    test_code("./Orders_mc2p1_spr2016/orders-leverage-3.csv")

## test cases from order's directory

#     test_code("./Orders/orders-leverage-1.csv")
#     test_code("./Orders/orders-leverage-2.csv")  # failing expected val : 1032955
#     test_code("./Orders/orders-leverage-3.csv")

#     start = time.time()
#     test_code("./Orders_mc2p1_spr2016/orders_large.csv")
#     end = time.time()
#     print (end-start)
     
 
        