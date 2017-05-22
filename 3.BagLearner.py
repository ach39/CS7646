# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 21:28:47 2017

    @author: ac104q
    @description : baglearner
	@date : Feb-19-2017
"""

import numpy as np
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
        
        bag_predY = np.zeros(points.shape[0])
        for i in range(self.bags):
            predY = self.learners[i].query(points)
            bag_predY += predY
        
        bag_predY = bag_predY/self.bags
        
        return(bag_predY)
        
        pass