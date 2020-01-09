# -*- coding: utf-8 -*-
"""
Created on Mar-29

    @author: ac104q
    @description : baglearner for PROJECT-3

"""

import numpy as np
import scipy.stats as scs
import pandas as pd

DBG=0

class BagLearner(object) :
    
    def __init__(self, learner, kwargs = {"leaf_size":1}, bags=10, boost=False, verbose=False):
        self.learner = learner
        self.args = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        
        self.N = 0
        self.learners=[]
        self.leaf_size = 1
        
        if kwargs.has_key("leaf_size"):
            self.leaf_size = kwargs['leaf_size']
        
        if(DBG) : print " Leaf size = " , self.leaf_size
        
        if(self.bags == 0):
            #print " Constructor : Warning - bag count is zero. Setting it to 1"
            self.bags =1
        

        # construct learners 
        for i in range(0,self.bags):
            
            self.learners.append(learner(**kwargs))
            #self.learners[i] = learner(**kwargs)
            #print "learner[] :",  self.learners[i]
            
    

    def author(self):
        return 'achauhan39'
        
    def addEvidence(self,dataX,dataY):
        self.N = dataX.shape[0]        
        sampled_idx = np.random.choice(range(0,self.N) ,size = self.N, replace=True) 
        trainX = dataX[sampled_idx, :]
        trainY = dataY[sampled_idx]
                
        self.learners[0].addEvidence(trainX, trainY) # training step 

        
        for i in range(1,self.bags):
            #print "creating Bag :" , i
            if(self.boost) :
                print "boosting not supported yet"
            else:
                sampled_idx = np.random.choice(range(0,self.N) ,size = self.N, replace=True) 
                trainX = dataX[sampled_idx, : ]
                trainY = dataY[sampled_idx]

                self.learners[i].addEvidence(trainX, trainY)
                
        if(DBG) : print "Baglearners len : " , len(self.learners)

    
    
    def query(self,points):
        
#        bag_predY = np.zeros(points.shape[0])       
#        for i in range(self.bags):
#            predY = self.learners[i].query(points)
#            bag_predY += predY
#        
#        bag_predY = bag_predY/self.bags
 #       return(bag_predY)
        
        #print "new bag learner "
        main = np.zeros(points.shape[0])
        for i in range(self.bags):
            predY = self.learners[i].query(points)
            if(i==0):
                main = predY
            else :
                main = np.column_stack((main,predY))
                
        df = pd.DataFrame(main)
        #print df.head(20)
        res = df.mode(axis=1)
        #print "res"
        #print res.head(20)
        res = res[0].values
        
        return(res)
        
        
        
        
        pass
    
    
#    a1=np.array([2,3,4,5])
#    a2=np.array([1,9,7,8])
#    a3=np.array([2,3,4,5])
#    
#    main=np.array([0,0,0,0])
#    main = np.column_stack((main,a1))
#    main = np.column_stack((main,a2))
#    main = np.column_stack((main,a3))
#    main
#    df = pd.DataFrame(main)
#    res = df.mode(axis=1)
#    res1 = res[0].values
#    print type(res1)
#    print res1
    
 
    
    
    
    