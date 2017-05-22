"""MC1-P2: Optimize a portfolio. Jan 27"""

#import matplotlib
#matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as sco

#import test_util_p2

#def min_func_sr(prices, allocs, rfr = 0.0, sf = 252.0):
#    cr,adr,sddr,sr = compute_portfolio_stats(prices, allocs, rfr, sf)
#    return -sr


def compute_portfolio_stats(prices_df, allocs, rfr = 0.0, sf = 252.0):
    
    '''
     Function : compute_portfolio_stats
     input    : price data frame, allocatoions, risk free rate of return, 
                sampling frequency
     outptut  : cumulative return , avg daily return, 
                std_dev of daily return , sharpe ratio , port_val
    '''
    
    normed = prices_df/prices_df.ix[0]
    port_val = (normed * allocs ).sum(axis=1)
        
    cr = (port_val[-1]/port_val[0]) -1      # Cummulative Return

    dr = (port_val/port_val.shift(1)) -1    # Daily Return
    dr.ix[0] = 0
    dr = dr[1:]                             # DR- Remove top row

    adr = dr.mean()                         # Average Daily Return
    sddr = dr.std()                         # STD of Daily Return
    dr_rfr_delta = dr-rfr
    sr = np.sqrt(sf) * (dr_rfr_delta.mean()/sddr)     # Sharpe Ratio

    return cr,adr,sddr,sr,port_val
    
    
        


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    
    ## ac : Error handling
    prices_all.fillna(method ='ffill')  
    prices_all.fillna(method ='bfill')
    
    prices = prices_all[syms]           # only portfolio symbols
    prices_SPY = prices_all['SPY']      # only SPY, for comparison later

     
    # find the allocations for the optimal portfolio
    
    num_of_syms = len(syms)
    init_guess = np.array(num_of_syms * [1.0/num_of_syms, ])
    bnds = tuple((0.0, 1.0) for x in range(num_of_syms))
    
    cons = ({'type': 'eq', 'fun' : lambda x : 1.0 - np.sum(x)})
    
    def min_func_sr( allocs = init_guess, prices_df = prices, rfr = 0.0, sf = 252.0):
        ''' Returns min Sharpe Ratio for a given portfolio '''
        cr,adr,sddr,sr, port_val = compute_portfolio_stats(prices, allocs, rfr, sf)
        return -sr
        
    result = sco.minimize(min_func_sr, init_guess, bounds=bnds , method='SLSQP', constraints =cons, options={'disp': True})                           
    allocs = result.x


    cr, adr, sddr, sr ,port_val  = compute_portfolio_stats(prices,allocs) # add code here to compute stats



    # Get daily portfolio value
    #port_val = prices_SPY # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp_norm = df_temp/df_temp.ix[0]
        plot_data(df_temp_norm, title="Daily Portfolio Value and SPY", xlabel="Date", ylabel="Normalised Price")
        pass

    return allocs, cr, adr, sddr, sr



def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols =  ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
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

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
    #test_util_p2.my_test_code()
