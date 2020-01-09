""" PROJECT -3  : Market simulator.
@author : achauhan39


PROJECT -3 

Important -  
	drop date 2011-06-15
	recompute if leverage > 1.5



"""


import pandas as pd
import numpy as np
import datetime as dt
import os
import time
import matplotlib.pyplot as plt

from util import get_data, plot_data

LEV_DBG = 0
DEBUG= 0
N=100


    
def author():
    return 'achauhan39'
    
    
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 100000):
    
    print "Processing orders from : ", orders_file
 
    #orders_file = "bm_order_in_xxx.csv"  ##***** remove it ******
    #start_val = 100000
    
    orders_df = pd.read_csv(orders_file,index_col='Date',parse_dates=True, 
                            usecols=['Date', 'Symbol', 'Order', 'Shares'],na_values=['nan'])
   
    #orders_df = pd.read_csv(orders_file,index_col='Date',parse_dates=True, na_values=['nan'])
           
   
    orders_df.sort_index(inplace=True)          #sort dates
    
    #print orders_df
    
    
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
    
#    if'2009-05-12' in price_df.index :   
#        price_df.drop(pd.Timestamp('2009-05-12'), inplace=True)  # drop Jun-15-2011 from price_df
    
    price_df['cash'] = np.ones(price_df.shape[0])            # Add Cash column
    
    if DEBUG : print price_df.head(n=N)
    #print price_df['2011-06-5' : '2011-06-20']
    
    """
        2. Create Trade data frame
    """
    trade_df = price_df.copy()
    trade_df.ix[:,:] = 0
#    cols = trade_df.columns.values
#    for c in cols :
#        trade_df[c] = np.zeros(trade_df.shape[0])
    
    #if DEBUG : print trade_df.head(n=N)
    
    #iterate over orders index 
    for date in price_df.index:
        if date in orders_df.index:
            sub_order = orders_df.ix[date :date]
            #print  date, sub_order.shape[0] , sub_order.shape
            #print sub_order
            
            for i in range(0,sub_order.shape[0]):
                sym = sub_order.ix[i,'Symbol']
                order = sub_order.ix[i,'Order']
                shares = sub_order.ix[i,'Shares']
                #print i, date, sym, order,shares
                
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
    #value_df = trade_df.copy()
    #value_df.ix[:,:] = 0  
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
#    lev_df = pd.DataFrame(index=holdings_df.index)
#    lev_df['leverage'] = (value_df.ix[: , :-1].abs().sum(axis=1)) / (value_df.ix[: , :-1].sum(axis=1) + value_df['cash'])
#    
#    ##df['Leverage'] = (df.ix[:,:-1].abs().sum(axis = 1)) / (df.ix[:,:-1].sum(axis = 1) + df['Cash'])
#    
#    lev_offender = lev_df[lev_df['leverage'] > 1000]
#    l_index = list(lev_offender.index)
#    if len(l_index)>0:
#        # get 1st offending index and drop if from orders_df
#        date = l_index[0]
#        
#        if LEV_DBG : print "Leverage exceeded :  dropped index "  , date  
#        
#        #is it last day ??
#        if date == ed :
#            print "leverage exceeded on last day" 
#            temp = orders_df.ix[date :date]
#            for i in range(0,temp.shape[0]):
#                temp.ix[i,'Shares'] = 0
#        else :
#            orders_df.drop(date, inplace=True)
#
#        #print orders_df
#        
#        orders_df.to_csv('modified_order.csv', sep=',')       
#        return compute_portvals( "modified_order.csv", start_val = 1000000)
        
    """
        7. Plot data
    """
    
#    buy_dates = orders_df.index[orders_df['Order']=='BUY']    
#    sell_dates = orders_df.index[orders_df['Order']=='SELL'] 
#    
#    plot = port_val_df.plot()    
#    ymin, ymax = plot.get_ylim()
#    plot.vlines(x=sell_dates, ymin=ymin, ymax=ymax-1,color='r')
#    plot.vlines(x=buy_dates, ymin=ymin, ymax=ymax-1,color='g')
    
    #plot normalised
#    price_SPY= get_benchmark()
#    #tmp_df = pd.concat([port_val_df,price_SPY],keys=['Portfolio','benchmark'], axis=1)
#    tmp_df = port_val_df
#    tmp_df['BM'] = price_SPY
#    tmp_df_norm = tmp_df/tmp_df.ix[0]
#    plot = tmp_df_norm.plot()
#    ymin, ymax = plot.get_ylim()
#    plot.vlines(x=sell_dates,ymin=ymin,ymax=ymax,color='r')
#    plot.vlines(x=buy_dates ,ymin=ymin,ymax=ymax,color='g')
    
    """
     8. test - join orders to port_val
    """
    tmp_port_val =  port_val_df.join(orders_df['Order']) 
    tmp_port_val.to_csv("tmp_port_val.csv")
    
    return port_val_df


#def get_benchmark() :
#    sd = dt.datetime(2011,1,1)
#    ed = dt.datetime(2011,12,31)
#    dates=pd.date_range(sd,ed)
#    price_SPY = get_data(['SPY'],dates)
#    price_SPY.plot()
#    return  price_SPY
#
#
#def test_diff_point():
#   df = pd.read_csv("test_diff_points.csv",index_col='Date',parse_dates=True)
# 
#   buy = df[df['act']=='BUY']
#   plt.scatter(x=buy.index, y=buy['bbp'] ,color='g')
#   plt.scatter(x=buy.index, y=buy['psma'],color='g')
#   
#   sell = df[df['act']=='SELL']
#   plt.scatter(x=sell.index, y=sell['bbp'] ,color='r')
#   plt.scatter(x=sell.index, y=sell['psma'],color='r')    
#       
#   nada = df[df['act']=='nada'] 
#   plt.scatter(x=nada.index, y=nada['bbp'] ,color='k')
#   plt.scatter(x=nada.index, y=nada['psma'],color='k')  
#   
#   plt.title("Features X1,X2 - Manual Strategy")
#   plt.ylabel("features")
#   plt.xticks(rotation=45)
#   plt.show()

    
    
    
def get_portfolio_stats(port_val, rfr=0.0 , sf=252.0, verbose=False, title=""):
    
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
    
    if(verbose==True):
        print "----------------------------------------------------"
        print "\n" ,title
        print "Cummulative_Return       : " , cr
        print "Stdev of Daily Returns   : " , sddr
        print "Mean of Daily Returns    : " , adr
        print "Sharpe Ratio             : " , sr
        print "----------------------------------------------------"
    
    return cr,adr,sddr,sr
  
    
    
    
def test_code(of , sv = 100000 ):

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
    return portvals


if __name__ == "__main__":

    print author()
    
    test=1
    
    if(test) :
        #order_file = "optimal_rule_based_orders.csv"
        
        order_file = "rule_based_orders.csv"
        
        bm          = test_code("bm_order_in.csv")
        port_val    = test_code(order_file)
    
        #bm          = test_code("bm_order_out.csv")
        #port_val    = test_code("test_order.csv")
        
        orders_df = pd.read_csv(order_file,index_col='Date',parse_dates=True, na_values=['nan']) 
        buy_dates = orders_df.index[orders_df['Order']=='BUY']    
        sell_dates = orders_df.index[orders_df['Order']=='SELL'] 
    
    
        #print bm.head()
        #print port_val.head()
        
        tmp_df=pd.DataFrame(index = bm.index)
        tmp_df['BM'] = bm
        tmp_df['portfolio'] = port_val
        tmp_df = tmp_df.fillna(method='bfill')
        tmp_df = tmp_df.fillna(method='ffill')
        #print tmp_df.head(30)
        
        #tmp_df_norm = tmp_df
        tmp_df_norm = tmp_df/tmp_df.ix[0]
        
        
        #print tmp_df.head()
        
        ax = tmp_df_norm['BM'].plot(label='benchmark',color='k',title="Rule based Strategy - APPL")
        tmp_df_norm['portfolio'].plot( ax=ax, label='portfolio',color='b')
        ymin, ymax = ax.get_ylim()
        ax.vlines(x=sell_dates,ymin=ymin,ymax=ymax,color='r')
        ax.vlines(x=buy_dates ,ymin=ymin,ymax=ymax,color='g')
        ax.set_ylim([ymin,ymax])
        plt.show()
    

    
        
        
 

  







        