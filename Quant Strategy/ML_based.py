# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 11:27:14 2017
@author: achauhan39
Version -Final( Apr-09)

"""
import os
import math
import pandas as pd
import numpy as np
import datetime as DT
import matplotlib.pyplot as plt

import RTLearner as rt
import BagLearner as bl

from util import get_data, plot_data
from marketsim import compute_portvals,get_portfolio_stats
from Indicators import get_features



DBG=0



TRAIN_SD     = DT.datetime(2008,01,01)
TRAIN_ED     = DT.datetime(2009,12,31)
TEST_SD = DT.datetime(2010,01,01)
TEST_ED = DT.datetime(2011,12,31)

train_data_csv  = "train_data.csv"
test_data_csv   = "test_data.csv"
train_order_csv = "ML_train_order.csv"
test_order_csv  = "ML_test_order.csv"

PLOT        =  0
START_CASH  = 100000            # starting cash
S_BLOCK     = 200               # allowed block of shares
DAYS_TO_WAIT = 21

BUY  = 1
SELL = -1
NADA = 0

##########################################################
# Run baglearner
# Pass the csv to run RandomTree Model with Bagging
# 
###########################################################
def run_bl(trainX,trainY,testX,testY,leaf=5,bag=20):

    #c_in[i], c_out[i] ,rmse_in[i], rmse_out[i] = test_RTLearner(trainX,trainY,testX, testY , leaf_size, test_type=3 , verbose=False)
    #c_in[i], c_out[i], rmse_in[i], rmse_out[i] = test_BagLearner(trainX,trainY,testX,testY, kwargs = {"leaf_size": 10}, bags=i*5 , boost=False, verbose=False)
    
    ## test Baglearner
    learner = bl.BagLearner(learner = rt.RTLearner , kwargs={"leaf_size": leaf} , bags=bag , boost = False , verbose=False)
    learner.addEvidence(trainX, trainY)
    
    #  In of sample
    predY_train = learner.query(trainX) # get the predictions
    rmse_in = math.sqrt(((trainY - predY_train) ** 2).sum()/trainY.shape[0])
    c_in = np.corrcoef(predY_train, y=trainY) 


    #df = pd.DataFrame(predY)
    #df.to_csv("predY_train.csv")
    #print "correct vs. incorrect prediction for train :" , (predY == trainY).sum() , (predY != trainY).sum() ,"|",  (predY == trainY).sum()/(trainY.shape[0] *1.0)
    
    if(DBG) : 
        print "\n In sample results for BagLearner"
        print "RMSE: ", rmse_in   
        print "corr: ", c_in[0,1]

   #  Out of sample
    predY_test = learner.query(testX) # get the predictions
    rmse_out = math.sqrt(((testY - predY_test) ** 2).sum()/testY.shape[0])
    c_out = np.corrcoef(predY_test, y=testY) 
    
    if(DBG) :
        print "\n Out of sample results for BagLearner"
        print "RMSE: ", rmse_out  
        print "corr: ", c_out[0,1]
  
    print leaf,bag , " : " , c_in[0,1],  c_out[0,1], rmse_in, rmse_out
    
    return (predY_train, predY_test)
    




##########################################################
# test_model : Run bagLearner on train and test data
# and get predictions 
# 
###########################################################
def test_model():

# 1.Read in Train data
    f = open(train_data_csv)
    if(DBG): print "\n USING TRAIN FILE : ", f
    train_data = np.array([map(float,s.strip().split(',')) for s in f.readlines()])
    #print train_data.shape
    f.close()
    
#2 .Read in Test data    
    f = open(test_data_csv)
    if(DBG): print "\n USING TEST FILE : ", f
    test_data = np.array([map(float,s.strip().split(',')) for s in f.readlines()])
    #print test_data.shape
    f.close()
    
#3. normalize trainX
#    NORMALIZE=0
#    if(NORMALIZE) : 
#        df = pd.read_csv(f,header=None)
#        norm_df= df.copy()
#        #norm_df[0] = (df[0]-df[0].mean()) / df[0].std()
#        #norm_df[1] = (df[1]-df[1].mean()) / df[1].std()
#        
#        # Todo : Normalize test and trains sepaarately
#        for i in range(norm_df.shape[1]) :
#            norm_df[i] = (df[i]-df[i].mean()) / df[i].std()
#        
#        norm_df.to_csv("Data/3_groups_1_norm.csv", index=False, header=False)
#        
#        inf = open("Data/3_groups_1_norm.csv")
#        print "\n USING NORM FILE : Data/3_groups_1_norm.csv "
#        data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
#        print data.shape
#        inf.close()
   
    
# 4.Run Bag learner

    # separate out training and testing data  
    trainX = train_data[:,0:-1]
    trainY = train_data[:,-1]
    
    testX  = test_data[:,0:-1]
    testY  = test_data[:,-1]
    
    if(DBG): print "train X & Y " , train_data.shape, trainX.shape , trainY.shape
    if(DBG): print "test X & Y " ,  test_data.shape, testX.shape , testY.shape
    
    #run_BagLearner(trainX,trainY,testX,testY, kwargs = {"leaf_size": 5}, bags=20 , boost=False, verbose=False)
    predY_train, predY_test = run_bl(trainX,trainY,testX,testY,5,20)
    
    if(DBG): print "correct vs. incorrect prediction for train :" , (predY_train == trainY).sum() , (predY_train != trainY).sum() , \
    "|",  (predY_train == trainY).sum()/(trainY.shape[0] *1.0)
    
    if(DBG): print "correct vs. incorrect prediction for test :" , (predY_test == testY).sum() , (predY_test != testY).sum() , \
    "|",  (predY_test == testY).sum()/(testY.shape[0] *1.0)
    
    return (predY_train, predY_test)
    
 
##########################################################
#  plot_port_bm : plot portfolio value against benchmark
# 
###########################################################
#def plot_port_bm(orders_df, port_df, bm_df, title ):

def plot_port_bm(df, port_df, bm_df, title ):
   
    #buy_dates  = orders_df.index[orders_df['Order'] == 'BUY']    
    #sell_dates = orders_df.index[orders_df['Order'] == 'SELL'] 
#    if (df['action']=='BUY') or (df['action']=='SELL') :
#        print df['action']

 
    buy_dates = df.index[df['action']=='BUY']    
    sell_dates = df.index[df['action']=='SELL'] 
    close_dates = df.index[df['action']=='CLOSE'] 
    
    tmp_df = pd.DataFrame(index = bm_df.index)
    tmp_df['BM'] = bm_df
    tmp_df['portfolio'] = port_df
     
    tmp_df = tmp_df.fillna(method='bfill')
    tmp_df = tmp_df.fillna(method='ffill')
   
    tmp_df_norm = tmp_df/tmp_df.ix[0]
    
    #tmp_df_norm = tmp_df_norm[ :120]  ##
    #print tmp_df_norm.head(10)
        
    ax = tmp_df_norm['BM'].plot(label='benchmark',color='k')
    tmp_df_norm['portfolio'].plot( ax=ax, label='portfolio',color='b')
    
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=sell_dates,ymin=ymin,ymax=ymax,color='r')
    ax.vlines(x=buy_dates ,ymin=ymin,ymax=ymax,color='g' )
    ax.vlines(x=close_dates ,ymin=ymin,ymax=ymax,color='k', linestyle=':') 
    ax.set_ylim([ymin,ymax])
    plt.title(title,color='m' )
    plt.ylabel("Normalised Price")
    plt.legend(loc='best')
    plt.show()
    
    print title
    cr,adr,sddr,sr               = get_portfolio_stats(tmp_df['portfolio'], verbose=True, title='portfolio') 
    cr_bm, adr_bm,sddr_bm ,sr_bm = get_portfolio_stats(tmp_df['BM'], verbose=True, title='Benchmark')
    

    
##########################################################
# add_label : Compuet Y label based on 21 day returns
# 
###########################################################    
#def add_label(df_feature):    
#    df = df_feature.copy()        
#    df['cr_21'] = (df['AAPL']/df['AAPL'].shift(21)) - 1    
#    df['dr'] = (df['AAPL']/df['AAPL'].shift(1)) - 1
#    df['adr_21'] = df['dr'].rolling(window=21, min_periods=21).mean()
#    
#    # drop first 21 rows as there will be no Y label for those days
#    df = df[21:]
#    # try1 :  
#    #df['label'] = np.where(df['cr_21'] > 2.0 , SELL, np.where(df['cr_21'] < -2.0 , BUY, 0))
#    
#    df['label'] = 0
#    df.loc[df['cr_21'] >  0.15, 'label'] = SELL
#    df.loc[df['cr_21'] < -0.25, 'label'] = BUY
#    return df  

# golden - +/-0.05
#def add_label(df_feature,lookback=14):    
#    df = df_feature.copy()        
#    df['cr_21'] = (df['AAPL'].shift(-21) / df['AAPL'])-1   
#    
#    #df['dr'] = (df['AAPL']/df['AAPL'].shift(1)) - 1
#    #df['adr_21'] = df['dr'].rolling(window=21, min_periods=21).mean()
#    
#    # drop first 21 rows as there will be no Y label for those days
#    df = df[lookback:]
#    df=df.fillna(method='bfill')
#    
#    df['label'] = 0
#    df.loc[df['cr_21'] < -0.05, 'label'] = SELL
#    df.loc[df['cr_21'] > 0.05, 'label'] = BUY  
#    return df 



def add_label(df_feature,lookback=14):    
    df = df_feature.copy()        
    df['cr_21'] = (df['AAPL'].shift(-21) / df['AAPL'])-1   
    
    # drop first 21 rows as there will be no Y label for those days
    df = df[lookback:]
    
    df = df.fillna(method='bfill')
    df['label'] = 0
    
    if(0) :
        median = df['cr_21'].median()
        std = 1.5 * df['cr_21'].std()   
        #Y[Y>(median+std)], Y[Y<(median-std)], Y[((Y<=(median+std)) & (Y>= (median-std)))] = 1,-1, 0
        df.loc[df['cr_21'] < (median-std), 'label'] = SELL
        df.loc[df['cr_21'] > (median+std), 'label'] = BUY      
        print "\n Median , Std : " , median, std , (median+std) , (median-std)
    else:
       df.loc[df['cr_21'] < -0.05, 'label'] = SELL
       df.loc[df['cr_21'] > 0.05, 'label'] = BUY  
    
    
    #df['cr_21'].plot()
    #plt.show()
    
    
    return df 
    
#def assignLabels(Y, median, std):
#    std = 1.1 * std;
#    Y[Y>(median+std)], Y[Y<(median-std)], Y[((Y<=(median+std)) & (Y>= (median-std)))] =1,-1, 0
#    #Y.plot()
#    #plt.show()
#    return Y


##########################################################
# build_ml_orders
#
###########################################################                 
def build_ml_orders(df):

    orders=[]
    holdings = 0
    wait_period = 21
          
    dates = df.index
    for dt in dates :    
        wait_period -= 1
        
        if(df.ix[dt,'predY'] == BUY) and (holdings < S_BLOCK) and (wait_period <= 0) :
            holdings += S_BLOCK
            df.ix[dt,'action'] = 'BUY'
            orders.append([dt,'AAPL','BUY', S_BLOCK])
            wait_period = 21
            
        if(df.ix[dt,'predY'] == SELL) and (holdings > -S_BLOCK) and (wait_period <= 0):
            holdings -= S_BLOCK
            df.ix[dt,'action'] = 'SELL'
            orders.append([dt,'AAPL','SELL', S_BLOCK])
            wait_period = 21
        
        if(wait_period == 0) :
            # must close postion
            if(holdings < 0):
              holdings = 0
              wait_period = 0
              df.ix[dt,'action'] = 'CLOSE'
              orders.append([dt,'AAPL','BUY', S_BLOCK,])  
              #print dt,"close postion BUY"
                      
            if(holdings > 0):
              holdings = 0
              wait_period = 0
              df.ix[dt,'action'] = 'CLOSE'
              orders.append([dt,'AAPL','SELL', S_BLOCK])
              #print dt,"close postion SELL"           


    #print orders
    
    #df_subset = df[ df['action'] != 'NADA']
    #df_orders = df_subset['action']
    
#    df_order=pd.DataFrame(index=df.index)
#    df_order['Symbol'] = 'AAPL'
#    df_order['Shares'] = S_BLOCK
#    df_order['Order']  = df['action']
#    df_order = df_order[ df_order['Order'] != 'NADA']
    
    #df_order=df_order.dropna(axis=0)
    
    df_order = pd.DataFrame(orders, columns=['Date','Symbol', 'Order', 'Shares'])
    df_order = df_order.set_index('Date')
    
    return df_order

def plot_scatter(filename , f1='bbp' ,f2='d_slow'):
   
#   filename = "scatter-in_sample.csv"
#   f1='bbp'
#   f2='d_slow'
   
   # Plot data based on Actual label
   df = pd.read_csv(filename)
   buy = df[df['label'] == BUY]
   sell = df[df['label'] == SELL]
   nada = df[df['label'] == NADA]
   
   plt.scatter(x=buy[f1],  y=buy[f2]  ,color='g' ,label='LONG')
   plt.scatter(x=sell[f1], y=sell[f2] ,color='r', label='SHORT')
   plt.scatter(x=nada[f1], y=nada[f2] ,color='k' ,label='NADA')
   plt.title("2. SSO vs. BB% - Training label", fontweight="bold" ,color='b')
   plt.xlabel('BB %', fontweight="bold")
   plt.ylabel(f2 ,fontweight="bold")
   plt.xlim(-1.5,1.5)
   plt.ylim(-1.5,1.5)
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   #plt.xticks(rotation=45)
   plt.show()
    
   # Plot data based on predicted label
   buy = df[df['predY'] == BUY]
   sell = df[df['predY'] == SELL]
   nada = df[df['predY'] == NADA]
   
   plt.scatter(x=buy[f1],  y=buy[f2]  ,color='g' ,label='LONG')
   plt.scatter(x=sell[f1], y=sell[f2] ,color='r', label='SHORT')
   plt.scatter(x=nada[f1], y=nada[f2] ,color='k' ,label='NADA')
   plt.title("3. SSO vs. BB% - Predicted label", fontweight="bold" ,color='b')
   plt.xlabel('BB %', fontweight="bold")
   plt.ylabel(f2 ,fontweight="bold")
   plt.xlim(-1.5,1.5)
   plt.ylim(-1.5,1.5)
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   plt.show()
    

##########################################################
# Q6 - plot out of sample data
#
###########################################################   
#def Q6(pv_rule):
def plot_Q6(pv_rule):
    
    pv_ml = compute_portvals("test_order_Q6.csv", start_val = START_CASH)
    bm_df = compute_portvals("bm_order_out.csv" , start_val = START_CASH)
        
    tmp_df = pd.DataFrame(index = bm_df.index)
    tmp_df['BM'] = bm_df
    tmp_df['pv_rule'] = pv_rule
    tmp_df['pv_ml'] = pv_ml 
    
    tmp_df = tmp_df.fillna(method='bfill')
    tmp_df = tmp_df.fillna(method='ffill')
    
    
    tmp_df_norm = tmp_df/tmp_df.ix[0]
        
    ax = tmp_df_norm['BM'].plot(label='benchmark',color='k')
    tmp_df_norm['pv_ml'].plot( ax=ax, label='ML strategy',color='g')
    tmp_df_norm['pv_rule'].plot( ax=ax, label='Manual strategy',color='b')
  
    plt.title("Out-of-Sample Performance " ,color='m' )
    plt.ylabel("Normalised Price")
    plt.legend(loc='best')
    plt.show()
    
    print "\n"
    cr_bm, adr_bm,sddr_bm ,sr_bm = get_portfolio_stats(tmp_df['BM'], verbose=True, title='Benchmark')
    cr,adr,sddr,sr               = get_portfolio_stats(tmp_df['pv_ml'], verbose=True, title='Out-of-Sample : ML Strategy') 
    cr,adr,sddr,sr               = get_portfolio_stats(tmp_df['pv_rule'], verbose=True, title='Out-of-Sample : Manual Strategy')

    
##########################################################
#
##########################################################
def init() :
    
    if os.path.isfile('bm_order_in.csv') == False:
        print 'Creating bm_order_in.csv.' 
        # 1/2/2008 - 12/31/2009
        orders=[]
        orders.append([ DT.datetime(2008,01,02) ,'AAPL','BUY', S_BLOCK])
        orders.append([ DT.datetime(2009,12,31) ,'AAPL','SELL', S_BLOCK])
        
        df_order = pd.DataFrame(orders, columns=['Date','Symbol', 'Order', 'Shares'])
        df_order = df_order.set_index('Date')
        df_order.to_csv('bm_order_in.csv', index_label='Date')
    
    
    if os.path.isfile('bm_order_out.csv') == False:
        print 'Creating bm_order_out.csv s'
        # 1/4/2010 - 12/30/2011
        orders=[]
        orders.append([ DT.datetime(2010,01,04) ,'AAPL','BUY', S_BLOCK])
        orders.append([ DT.datetime(2011,12,30) ,'AAPL','SELL', S_BLOCK])
        
        df_order = pd.DataFrame(orders, columns=['Date','Symbol', 'Order', 'Shares'])
        df_order = df_order.set_index('Date')
        df_order.to_csv('bm_order_out.csv', index_label='Date')    


   
##########################################################
# MAIN
#
###########################################################    
def run_ML_trader(scatter=False, out=False):
    symbols = ['AAPL']
    lookback = 14
    features = ['AAPL','bbp','psma','roc','d_slow' ]  #,'rel_p'
    init()
      
 # 1.PREPARE TRAIN DATA 
    
    sd = DT.datetime(2008,01,01)
    ed = DT.datetime(2009,12,31)
    
    #ed = DT.datetime(2008,05,30)    

    df_train = get_features(sd,ed,symbols)  
    train_ss = df_train.ix[:,features]  
    train_ss = add_label(train_ss ,lookback)
    train_ss.to_csv("train_ss.csv")
    #print train_ss.head()
    
    #create train data csv for BagLearner
    cols = ['bbp','psma','roc','d_slow','label']
    tmp_df = train_ss.ix[ : , cols ]  
    # normalise
    tmp_df['bbp']   = (tmp_df['bbp']-tmp_df['bbp'].mean())/tmp_df['bbp'].std()
    tmp_df['psma']  = (tmp_df['psma']-tmp_df['psma'].mean())/tmp_df['psma'].std()
    tmp_df['roc']   = (tmp_df['roc']-tmp_df['roc'].mean())/tmp_df['roc'].std()
    tmp_df['d_slow']   = (tmp_df['d_slow']-tmp_df['d_slow'].mean())/tmp_df['d_slow'].std()
    
    tmp_df.to_csv(train_data_csv, index=False, header = False)
    #print "writing to ", train_data_csv
    

# 2.PREPARE TEST DATA 
    sd = DT.datetime(2010,01,01)
    ed = DT.datetime(2011,12,31)
    
    #ed = DT.datetime(2008,05,30)    
    
    df_test = get_features(sd,ed,symbols)    
    test_ss = df_test.ix[:,features]  
    test_ss = add_label(test_ss,lookback) 
    test_ss.to_csv("test_ss.csv")
    
    #print test_ss.head()
    
    #create Test data csv for BagLearner
    tmp_df_2 = test_ss.ix[ : , cols ]  
    #normalise
    tmp_df_2['bbp']   = (tmp_df_2['bbp']-tmp_df_2['bbp'].mean())/tmp_df_2['bbp'].std()
    tmp_df_2['psma']  = (tmp_df_2['psma']-tmp_df_2['psma'].mean())/tmp_df_2['psma'].std()
    tmp_df_2['roc']   = (tmp_df_2['roc']-tmp_df_2['roc'].mean())/tmp_df_2['roc'].std()
    tmp_df_2['d_slow']   = (tmp_df_2['d_slow']-tmp_df_2['d_slow'].mean())/tmp_df_2['d_slow'].std()

    tmp_df_2.to_csv(test_data_csv, index=False, header = False )
    #print "writing to ", test_data_csv
    
    
#3. TEST MODEL
    predY_train, predY_test = test_model()
    
    train_ss['predY'] = predY_train
    tmp_df['predY'] = predY_train
    tmp_df.to_csv("scatter-in_sample.csv")
    if(scatter) : plot_scatter("scatter-in_sample.csv" , 'bbp','d_slow')
    
    test_ss['predY']  = predY_test
    

#4. Build order file
    
    # orders of training period
    train_ss['action'] = 'NADA'
    orders_df_train = build_ml_orders(train_ss)    
    orders_df_train.to_csv(train_order_csv, index_label='Date')
    port_val_train = compute_portvals(train_order_csv, start_val = START_CASH)  

#    # orders of Test period
    test_ss['action'] = 'NADA'
    orders_df_test = build_ml_orders(test_ss)    
    orders_df_test.to_csv(test_order_csv, index_label='Date')
    port_val_test = compute_portvals(test_order_csv, start_val = START_CASH)  


#5. plot data
    
    bm_df = compute_portvals( "bm_order_in.csv" , start_val = START_CASH)
    #plot_port_bm(orders_df_train , port_val_train , bm_df ,"ML strategy : Train period") 
    plot_port_bm(train_ss , port_val_train , bm_df ,"ML strategy : Train period") 
    train_ss.to_csv("tmp.csv")
    
    bm_df = compute_portvals( "bm_order_out.csv" , start_val = START_CASH)
    #plot_port_bm(orders_df_test ,  port_val_test  , bm_df, "ML strategy : Test Period")
    if(out) : plot_port_bm(test_ss ,  port_val_test  , bm_df, "ML strategy : Test Period")
    
    

    
if __name__ == "__main__" :
    run_ML_trader()
    
    

















##########################################################
# TEST CODE
#
###########################################################

def test_ml_data():
    f = "Data/3_groups_1.csv"
    #f = "ml_orders.csv"
    
    inf = open(f)
    print "\n USING FILE : ", f,  "\n"
    
#    if(f == 'Data/3_groups_2.csv'):
#        #it has header
#        data = np.array([map(float, s.strip().split(',')[1:]) for s in inf.readlines()[1:]])
#        print data.shape
#    else:    
#        data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])   

    data1 = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    print data1.shape
    inf.close()
    
    #normalize trainX
    NORMALIZE=1
    if(NORMALIZE) : 
        df = pd.read_csv(f,header=None)
        norm_df= df.copy()
        #norm_df[0] = (df[0]-df[0].mean()) / df[0].std()
        #norm_df[1] = (df[1]-df[1].mean()) / df[1].std()
        
        # Todo : Normalize test and trains sepaarately
        for i in range(norm_df.shape[1]) :
            norm_df[i] = (df[i]-df[i].mean()) / df[i].std()
        
        norm_df.to_csv("Data/3_groups_1_norm.csv", index=False, header=False)
        
        inf = open("Data/3_groups_1_norm.csv")
        print "\n USING NORM FILE : Data/3_groups_1_norm.csv "
        data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
        print data.shape
        inf.close()
   
    

   # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    #test_rows = data.shape[0] - train_rows

    # separate out training and testing data  - Todo : uncomment this
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX  = data[train_rows:,0:-1]
    testY  = data[train_rows:,-1]
    
    #run_BagLearner(trainX,trainY,testX,testY, kwargs = {"leaf_size": 5}, bags=20 , boost=False, verbose=False)
    run_bl(trainX,trainY,testX,testY,5,20)
    
    
##########################################################
# 
###########################################################

def run_BagLearner(trainX,trainY,testX,testY, kwargs,bags,boost=0, verbose=False):
    #print "Bag = ", bags
     
    ## test Baglearner
    learner = bl.BagLearner(learner = rt.RTLearner , kwargs=kwargs , bags=bags , boost = boost , verbose=verbose)
    print learner.author()
    learner.addEvidence(trainX, trainY)
    
    c_in  = -1
    c_out = -1
    predY = learner.query(trainX) # get the predictions
    ## save to csv  
    #print type(predY)
    #print predY
    df = pd.DataFrame(predY)
    df.to_csv("predY.csv")
    
    rmse_in = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    c_in = np.corrcoef(predY, y=trainY) 
    verbose=0
    if(verbose) : 
        print "\n In sample results for BagLearner"
        print "RMSE: ", rmse_in   
        print "corr: ", c_in[0,1]

   #  out of sample
    predY = learner.query(testX) # get the predictions
    rmse_out = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c_out = np.corrcoef(predY, y=testY) 
    if(verbose) :
        print "\n Out of sample results for BagLearner"
        print "RMSE: ", rmse_out  
        print "corr: ", c_out[0,1]
    
    print bags , " : " , c_in[0,1],  c_out[0,1], rmse_in, rmse_out
    return (c_in[0,1],c_out[0,1], rmse_in, rmse_out)
    
    #return (-1,-1)