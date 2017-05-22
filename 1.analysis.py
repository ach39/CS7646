"""MC1-P1: Analyze a portfolio.Jan-20 """

#import matplotlib
#matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

#import ac_test_util


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


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    ## ac : Error handling
    prices_all.fillna(method ='ffill')  
    prices_all.fillna(method ='bfill')
    if sum(allocs) != 1.0 :
            print "Error sum of alloc != 1 ", sum(allocs), allocs
            allocs = list(allocs/np.sum(allocs))
    if(len(syms) != len(allocs)):
        print "Error : Symbol/Allocation length mismatch"
        
        
    prices = prices_all[syms]           # only portfolio symbols
    prices_SPY = prices_all['SPY']      # only SPY, for comparison later
    

    # Get daily portfolio value
    normed = prices/prices.ix[0]
    alloced = normed * allocs
    pos_val = alloced * sv
    port_val = pos_val.sum(axis=1)  # sum of rows


    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr  = get_portfolio_stats(port_val,rfr,sf)
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp_norm = df_temp/df_temp.ix[0]
        #print df_temp.head(10)
        #print df_temp_norm.head(10)
       
#        ax = df_temp_norm['SPY'].plot(title="Daily Portfolio Value and SPY")
#        df_temp_norm['Portfolio'].plot(label='Portfolio', ax=ax)
#       
#        ax.set_xlabel("Date")
#        ax.set_ylabel("Normalised Price")
#        ax.legend(loc="upper left")
#       
#        #plt.show()       
#        plt.savefig('output/test.png')
#        plt.close()
        plot_data(df_temp_norm, title="Daily Portfolio Value and SPY", xlabel="Date", ylabel="Normalised Price")
 

    # Add code here to properly compute end value  - ac: not done ??
    ev = port_val[-1]

    #print sr,sddr,adr,cr, ev

    return cr, adr, sddr, sr, ev




def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    start_date = dt.datetime(2010,01,01)
    end_date = dt.datetime(2010,12,31)
    symbols=['AXP', 'HPQ']
    allocations =[0.0, 1.0]
    start=1000000
    risk_free_rate=0.001
    sample_freq=252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
	rfr = risk_free_rate, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    print " End val :" , ev



if __name__ == "__main__":
    test_code()
    #ac_test_util.my_test_code()



