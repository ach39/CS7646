"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

@author : achauhan39
@date : Apr-09

# Apr-21 : submission
# Apr-24 v2 : Add check on train_time (9:51pm)

"""

import datetime as dt
import time
import QLearner as ql
import pandas as pd
import util as ut
import os
import numpy as np


buffet=1
DBG=0
DBG2 = 0


S_BLOCK= 200
TRAIN_ITERATIONS = 32
lookback = 13
        
#Actions: 0 = BUY, 1 = SELL, 2 = NOTHING  - My orig settings
NADA = 0
SELL = 1
BUY = 2

#positions: 0 = CASH, 1 = SHORT, 2 = LONG
CASH = 0
SHORT = 1
LONG = 2

FEATURE_CNT = 3  
STEPS = 10      
        
class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.thresholds = np.zeros((STEPS , FEATURE_CNT))
        self.sv = 100000
        self.ql = ql.QLearner(num_states=3000, num_actions=3, rar=0.5, radr=0.99,dyna=40)

        #print "StrategyLearner ...init"

    def author(self):
        return 'achauhan39'

        
    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2006,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 100000): 
        
        self.symbol = symbol
        self.sd = sd
        self.ed = ed
        self.sv = sv

        self.learner = ql.QLearner()

        features_df = self.getFeatures(sd,ed,[symbol])        
        self.compute_thresholds(features_df)
        features_df = self.add_state(features_df)
        #features_df.to_csv("get_features_discretized.csv")

        self.trainQ(features_df)
        

    def trainQ(self,df_features):  
        
        df = df_features.copy()
        
        st = time.time()
        for i in range (0, TRAIN_ITERATIONS) :
            
            date=0
            cum_reward = 0
            df['pos'] = CASH
            if(DBG) : 
                df['action'] = 0
                df['reward'] = 0
                df['share'] = 0
                df['st_Q'] = 0
           
            
            # discritize feature values and derive state
            state = df.ix[date,'state'] + (df.ix[date-1,'pos'] *1000)  #cur_pos            
            action = self.ql.querysetstate(int(state))
            
            if(DBG):
                df.ix[date,'action']= action
                df.ix[date,'st_Q']= state
            
            #set current postion based on action returne by Q learner
            if(action == BUY) : 
                df.ix[date,'pos'] = LONG
                if(DBG): df.ix[date, 'share'] = S_BLOCK
                
            if(action == SELL) : 
                df.ix[date,'pos'] = SHORT
                if(DBG): df.ix[date, 'share'] = -S_BLOCK
            
            
            #loop through remaining days
            for date in range(1, df.shape[0]):
                reward = 0
                
                #compute reward based on % chnage in stock price 
                #dr = (df.ix[date,self.symbol] /df.ix[date-1, self.symbol]) -1
                if(action == NADA ):
                    reward = 0
                else:
                    if(df.ix[date-1,'pos'] == LONG) : reward = df.ix[date,'dr']
                    if(df.ix[date-1,'pos'] == SHORT) : reward = df.ix[date,'dr'] * -1

                #Pass state and reward from previous action to learner
                state = df.ix[date,'state'] + (df.ix[date-1,'pos'] * 1000) #cur_pos
                action = self.ql.query(int(state), reward)
                
                #print df.ix[date,'state'], "|" , df.ix[date-1,'pos'] , "|", state
                
                if(DBG):
                    df.ix[date,'action']= action
                    df.ix[date,'st_Q']= state
                    df.ix[date,'reward']= reward
                   
                df.ix[date,'pos'] = df.ix[date-1,'pos']
     
                if(action == BUY) and (df.ix[date-1,'pos'] != LONG) : 
                    df.ix[date,'pos'] = LONG
                    if(DBG): df.ix[date, 'share'] = S_BLOCK*2
                
                if(action == SELL) and (df.ix[date-1,'pos'] != SHORT) : 
                    df.ix[date,'pos'] = SHORT
                    if(DBG): df.ix[date, 'share'] = -S_BLOCK*2
                    
                cum_reward += reward
                
            end = time.time()
            if(end-st >=24) : break

            #print "Iteration = " , i , "Cum_Ret =",  cum_reward , "Time = ", end-st 
            
        #print (Cum_Ret =",  cum_reward )
        if(DBG):
            df.to_csv("train_Q.csv")
#            tmp = pd.DataFrame(index=df.index)
#            #tmp['tmp'] = df['share']
#            
#            tmp['Shares'] = 200            
#            tmp['Symbol'] = self.symbol
#            tmp['Order'] = np.where(df['share']>0 , 'BUY', 'SELL')
#            tmp = tmp[df['share'] !=0]
#            tmp.to_csv("train_order.csv",index_label='Date')
          
        return df
     
     
     # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2010,12,31), \
        sv = 100000):
            
        self.symbol = symbol
        self.sd = sd
        self.ed = ed
        self.sv = sv
        
        features_df = self.getFeatures(sd,ed,[symbol])
        features_df = self.add_state(features_df)
        trades = self.testQ(features_df)
        
        return trades


    def testQ(self,df):
        trade_df = pd.DataFrame(index=df.index, columns=['shares'])
        trade_df['shares'] = 0
        holdings=0
        cur_pos = CASH #0
        
#        if(DBG2) : 
#            tmp_df = df.copy()
#            tmp_df['state'] = 0
#            tmp_df['action'] = NADA

        #for date in df.index :
        for date in range(0, df.shape[0]):

            state = df.ix[date, 'state'] + cur_pos * 1000   #cur_pos           
            action = self.ql.querysetstate(int(state))
            
#            if(DBG2) : 
#                tmp_df.ix[date,'state'] = state
#                tmp_df.ix[date,'action'] = action

            if(cur_pos == CASH) :
                if(action == BUY) :
                    trade_df.ix[date, 'shares'] = S_BLOCK
                    holdings = S_BLOCK
                if (action == SELL) :
                    trade_df.ix[date, 'shares'] = -S_BLOCK
                    holdings = -S_BLOCK
            else :
                  #already had a position opened             
                if action == SELL and cur_pos == LONG:
                    trade_df.ix[date, 'shares'] = -S_BLOCK *2
                    holdings = -S_BLOCK
     
                if action == BUY and cur_pos == SHORT :
                    trade_df.ix[date, 'shares'] = S_BLOCK *2
                    holdings = S_BLOCK
            
            if holdings == S_BLOCK:
                cur_pos = LONG
            elif holdings == -S_BLOCK:
                cur_pos = SHORT
            else:
                cur_pos = CASH   #0                                      
  
#        if(DBG2) : 
#            tmp_df['shares'] = trade_df['shares']
#            tmp_df.to_csv("test_Q_debug.csv")
        
        return trade_df
  
   ## Utility Functions ###
  
    def compute_thresholds(self,df):
        '''
            0 - bbp
            1 - psma
            2 - d_slow
        ''' 
        df_bbp = df['bbp'].copy()
        df_psma = df['psma'].copy()
        df_dslow = df['d_slow'].copy()
        
        df_bbp.sort_values(inplace=True)
        df_psma.sort_values(inplace=True)
        df_dslow.sort_values(inplace=True)
        
        stepsize = df.shape[0]/STEPS
        for i in range(0, STEPS) :
            idx = (i+1)*stepsize
            self.thresholds[i,0] = df_bbp[idx]
            self.thresholds[i,1] = df_psma[idx]
            self.thresholds[i,2] = df_dslow[idx]
            if(DBG) : print i , stepsize,  idx

        if(DBG) : print self.thresholds
        
               
    def add_state(self, df_features): 
        df = df_features.copy()
        df['bbp_disc'] = 0
        df['psma_disc'] = 0
        df['dslow_disc'] = 0
        df['state']=0
        
        for date in range(0,df.shape[0]):        
            
            for i in range(0, STEPS):
                if df.ix[date,'bbp'] <= self.thresholds[i,0]:
                    df.ix[date,'bbp_disc'] += i
                    break
                if(i>=9) : df.ix[date,'bbp_disc'] = 9    
                   
            for i in range(0, STEPS):
                if df.ix[date, 'psma'] <= self.thresholds[i,1]:
                    df.ix[date,'psma_disc'] += i
                    break
                if(i>=9) : df.ix[date,'psma_disc'] = 9
    
            
            for i in range(0, STEPS):
                if df.ix[date, 'd_slow'] <= self.thresholds[i,2]:
                    df.ix[date,'dslow_disc'] += i
                    break
                if(i>=9) : df.ix[date,'dslow_disc'] = 9        
    
        #df['state'] = df['bbp_disc'] + df['psma_disc']*10 + df['dslow_disc']*100
        
        df['state'] = df['bbp_disc']*100 + df['psma_disc']*10 + df['dslow_disc']
        
        
#        if self.symbol == 'ML4T-220' :
#            print "WRANING : excluding d_slow for ," , self.symbol
#            df['state'] = df['bbp_disc']*100 + df['psma_disc']*10 + 1
            
        
        #print "typeof()" , type(df['state'])
        #df['state'] = df['state'].astype('int')
        
        return df 
        
        
    def getFeatures(self,sd,ed,symbol,norm=False):
        
        sd_org = sd
        sd = sd + dt.timedelta(days= -(lookback+15))
        
        dates = pd.date_range(sd,ed)
        df = ut.get_data(symbol,dates)
        
        #print "get_data  ...", symbol

                
        df = df.dropna(axis=0)
        
        #df.fillna(method ='ffill',inplace=True)  
        #df.fillna(method ='bfill',inplace=True)

        price = df[symbol]
        if(buffet) :
            sma   = pd.rolling_mean(price,window=lookback,min_periods=lookback)
            r_std = pd.rolling_std(price,window=lookback,min_periods=lookback)
        else :
            sma   = price.rolling(window=lookback,min_periods=lookback).mean()
            r_std = price.rolling(window=lookback,min_periods=lookback).std()
        
        bb_upper = sma + (2*r_std)
        bb_lower = sma - (2*r_std)
        
#        if(1) :
#            df['sma'] = sma
#            df['bb_upper'] = bb_upper
#            df['bb_lower'] = bb_lower
        
        df['bbp'] = (price - bb_lower)/(bb_upper - bb_lower)
        df['psma'] = price/sma
        #df['roc'] = ( (df[symbol]/df[symbol].shift(lookback-1)) - 1 )
        
        #compute DR
        normed = price/price.ix[0]
        daily_rets = (normed/normed.shift(1))-1
        daily_rets = daily_rets[1:]
        df['dr'] = daily_rets
        #df['dr_std'] = daily_rets.rolling(window=lookback,min_periods=lookback).std()
        
        SSO=1
        if(SSO) :        
            df_so = pd.DataFrame(index=df.index)
        
            #filename = os.path.join(os.path.join("..", "data"), "{}.csv".format(str(symbol[0])))
            
            
            filename = ut.symbol_to_path(symbol[0])
            #print "reading file ...", filename
            
            df_temp = pd.read_csv(filename,index_col='Date', parse_dates=True, na_values=['nan'])
            df_so = df_so.join(df_temp)
            
            ''' 
                Fast stochastic calculation
                    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
                    %D = 3-day SMA of %K
            '''
            if(buffet) :
                low_min  = pd.rolling_min(df_so['Low'] ,window=lookback)
                high_max = pd.rolling_max(df_so['High'],window=lookback)
            else :
                low_min  = df_so['Low'].rolling(window=lookback).min()
                high_max = df_so['High'].rolling(window=lookback).max()
            
            #df_so['low_min'] = low_min
            #df_so['high_max'] = high_max
            df_so['k_fast'] = (df_so['Adj Close'] - low_min)/(high_max - low_min) * 100
            
            if(buffet) :
                df_so['d_fast'] = pd.rolling_mean(df_so['k_fast'], window=3,min_periods=3)
            else :
                df_so['d_fast'] = df_so['k_fast'].rolling(window=3,min_periods=3).mean()
            
            
            """ 
                Slow stochastic calculation
                    %K = %D of fast stochastic
                    %D = 3-day SMA of %K
            """
            #df_so['k_slow'] = df_so['d_fast']
            #k_slow is same as d_fast
            if(buffet) :
                df_so['d_slow'] = pd.rolling_mean(df_so['d_fast'],window=3,min_periods=3)
            else :
                df_so['d_slow'] = df_so['d_fast'].rolling(window=3,min_periods=3).mean()
            
            #df_so.to_csv("df_so.csv")
            #df.to_csv("df.csv")
            
            df = df.join(df_so['d_slow'])
    
        
        #df.to_csv("get_features_1.csv")
        #df_so.to_csv("SSO.csv")
                
        df= df.sort_index()
        df = df[sd_org:]
        #df.to_csv("get_features_2.csv")
        
        return df
        
       
if __name__=="__main__":
    print "One does not simply think up a strategy"
        
    sd = dt.datetime(2008,01,01)
    ed = dt.datetime(2009,12,31)
    symbols = ['GOOG']  
    
    #learner = StrategyLearner(verbose = False)
    #df = learner.getFeatures(sd,ed,symbols) 









































###############################################################################################

 
#    def get_state(self, bbp, psma, dslow):
#        state = 0
#        print "In get_state : " , bbp, psma, dslow
#        
#        for i in range(0, STEPS):
#            if bbp <= self.thresholds[i,0]:
#                state += i
#                break
#        #if(i==9) : state += 9    
#        print "bbp disc_val : ", i , "state=", state
#        
#        for i in range(0, STEPS):
#            if psma <= self.thresholds[i,1]:
#                state += i * 10
#                break
#        #if(i==9) : state += 9*10
#        print "psma disc_val : ", i , "state=", state
#        
#        for i in range(0, STEPS):
#            if dslow <= self.thresholds[i,2]:
#                state += i * 100
#                break
#        #if(i==9) : state += 9*100        
#        print "dslow disc_val : ", i , "state=", state
#        
#        print state
#        return state 

#    def trainQ(self,df):
#        for i in range (0, TRAIN_ITERATIONS) :
#            st = time.time()
#            pv_df = self.init_pv_df(df)
#            date=0
#            cur_pos=0
#            
#            #state = self.get_state(bbp = df.ix[date,'bbp'], psma= df.ix[date,'psma'] , dslow= df.ix[date,'d_slow']) + cur_pos * 1000
#            state = df.ix[date,'state'] + cur_pos * 1000
#            
#            action = self.ql.querysetstate(state)
#            
#            for date in range(1, df.shape[0]):
#                pv_df = self.execute(date, pv_df,action)
#                cur_pos = self.get_cur_pos(pv_df, date)
#                reward = (pv_df.ix[date, 'port_val']/ pv_df.ix[date-1 ,'port_val']) -1
#            
#                #state = self.get_state( bbp = df.ix[date,'bbp'], psma= df.ix[date,'psma'] , dslow= df.ix[date,'d_slow']) + cur_pos * 1000
#                state = df.ix[date,'state'] + cur_pos * 1000
#                action = self.ql.query(state, reward)
#                
#            cum_ret = (pv_df.ix[-1,'port_val']/ pv_df.ix[0,'port_val'] )-1
#            end = time.time()
#            print "Iteration = " , i , "Cum_Ret =",  cum_ret , "Time = ", end-st ,"|", pv_df.ix[-1,'port_val'] , pv_df.ix[0,'port_val']
#          
#        print "End of training : Final CumRet =" ,  cum_ret 

#    def init_pv_df(self, df_features):
#        pv_df = pd.DataFrame(index = df_features.index ,columns=['price','share','cash','port_val'])
#        pv_df['cash']       = self.sv
#        pv_df['port_val']   = self.sv
#        pv_df['share']      = 0
#        pv_df['price']      = df_features.ix[:,self.symbol]
#        return pv_df
#        
#    
#    def get_cur_pos(self, pv_df,date):
#        if pv_df.ix[date, 'share'] == 0: 
#            cur_pos = 0
#        elif pv_df.ix[date, 'share'] < 0: 
#            cur_pos = SHORT
#        else: #long position
#            cur_pos = LONG
#        return cur_pos
#        
#        
#    def execute(self, date,pv_df, action):
#        
##        if(pv_df.ix[date,'share'] == 0):
##            pv_df.ix[date, 'share'] = S_BLOCK
#        
#        #cur_pos = self.get_cur_pos(pv_df, date)
#        
#        if (action == BUY) and (pv_df.ix[date,'share'] <= 0) :  # short or none position
#            pv_df.ix[date:, 'share'] += S_BLOCK
#            pv_df.ix[date:, 'cash'] = pv_df.ix[date, 'cash'] - pv_df.ix[date,'price'] * S_BLOCK
#        
#        
#        if (action == SELL) and (pv_df.ix[date,'share'] >= 0) :  # long or none position 
#            pv_df.ix[date:, 'share'] -= S_BLOCK
#            pv_df.ix[date:, 'cash'] = pv_df.ix[date, 'cash'] + pv_df.ix[date,'price'] * S_BLOCK
#                
#   
#        #update portfolio value and extend into the future
#        pv_df.ix[date:, 'port_val'] = pv_df.ix[date, 'cash'] + (pv_df.ix[date,'share'] * pv_df.ix[date, 'price'])
#        return pv_df


#def test_dates():
#    sd = dt.datetime(2008,01,01)
#    ed = dt.datetime(2008,1,20)
#    
#    sd_org = sd
#    sd = sd + dt.timedelta(days=-30)
#    dates = pd.date_range(sd,ed)
#    df_tmp = pd.DataFrame(index=dates)
#    df_tmp['price'] = 0
#    print df_tmp.head(10)
#    
#    df_tmp2 = df_tmp[sd_org:]
#    print df_tmp2.head(10)



###############################################################################################
# train_Q

            #set current postion based on action returne by Q learner
#            if(action == BUY) : 
#                cur_pos = LONG
#                df.ix[date, 'share'] = S_BLOCK
#                
#            if(action == SELL) : 
#                cur_pos = SHORT
#                df.ix[date, 'share'] = -S_BLOCK
            
            #loop through remaining days
#            for date in range(1, df.shape[0]):
#                reward = 0
#                
#                #compute reward based on % chnage in stock price 
#                #dr = (df.ix[date,self.symbol] /df.ix[date-1, self.symbol]) -1
#                if(action == NADA ):
#                    reward =0
#                else:
#                    if(cur_pos == LONG) : reward = df.ix[date,'dr']
#                    if(cur_pos == SHORT) : reward = df.ix[date,'dr'] * -1
#
#                #Pass state and reward from previous action to learner
#                state = df.ix[date,'state'] + cur_pos*1000
#                action = self.ql.query(state, reward)
#                
#                df.ix[date,'action']= action
#                df.ix[date,'st_Q']= state
#                df.ix[date,'reward']= reward
#            
#                # ??? -- need confirmation ????
#                if(action == BUY) and (cur_pos != LONG) : 
#                    cur_pos = LONG
#                    df.ix[date, 'share'] = S_BLOCK
#                if(action == SELL) and (cur_pos != SHORT) : 
#                    cur_pos = SHORT
#                    df.ix[date, 'share'] = -S_BLOCK
            
            
            
        
        
        
#    def disc_col(self, df ,col_name, grp_cnt=10):
#        df = df.sort_values([col_name])
#        new_col =col_name +'_disc'
#        
#        sz = df.shape[0]/grp_cnt
#        
#        st, end = 0 ,sz
#        for i in range(0,grp_cnt):
#            df.ix[st:end, new_col] = i+1
#            st = end
#            end = end+sz
#		#store threshold
#		#self.threshold[i, col_name] = df.ix[end,col_name]
#			
#        df[new_col].fillna(method='ffill' ,inplace=True)    
#        return df
#    
#    def discretize(self, df):
#        grp_cnt = 10
#        df = self.disc_col(df, 'bbp' , grp_cnt )
#        df = self.disc_col(df, 'psma' , grp_cnt )
#        df = self.disc_col(df, 'd_slow' , grp_cnt )
#        
#        df= df.sort_index()
#        df['state'] = df['bbp_disc'] + df['psma_disc']*10 + df['d_slow_disc'] *100
#        #df['reward'] = (df['Price']/df['Price'].shift(1)) -1
#        
#        return df


#test policy output

        # here we build a fake set of trades
        # your code should return the same sort of data
#        sd=dt.datetime(2009,1,1)
#        ed=dt.datetime(2009,2,1)
#        symbol="IBM"
#        dates = pd.date_range(sd, ed)
#        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
#        trades = prices_all[[symbol,]]  # only portfolio symbols
#        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
#        trades.values[:,:] = 0 # set them all to nothing
#        trades.values[3,:] = 500 # add a BUY at the 4th date
#        trades.values[5,:] = -500 # add a SELL at the 6th date 
#        trades.values[6,:] = -500 # add a SELL at the 7th date 
#        trades.values[8,:] = 1000 # add a BUY at the 9th date
#        
#        if self.verbose: print type(trades) # it better be a DataFrame!
#        if self.verbose: print trades
#        if self.verbose: print prices_all
        
        
        
                # example usage of the old backward compatible util function
#        syms=[symbol]
#        dates = pd.date_range(sd, ed)
#        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
#        prices = prices_all[syms]  # only portfolio symbols
#        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
#        if self.verbose: print prices
#  
#        # example use with new colname 
#        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
#        volume = volume_all[syms]  # only portfolio symbols
#        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
#        if self.verbose: print volume
        
        
# compute Thresholds        
#        df = df_features.copy()
#        stepsize = df.shape[0]/STEPS
#        
#        df = df.sort_values(['bbp'])   
#        for i in range(0, STEPS) :
#            self.thresholds[i,0] = df.ix[(i+1)*stepsize, 'bbp']
#        
#        df = df.sort_values(['psma'])
#        for i in range(0, STEPS) :
#            self.thresholds[i,1] = df.ix[(i+1)*stepsize, 'psma']
#
#        df = df.sort_values(['d_slow'])
#        for i in range(0, STEPS) :
#            self.thresholds[i,2] = df.ix[(i+1)*stepsize, 'd_slow']
        
        
#BUY = 0
#SELL = 1
#NADA = 2       