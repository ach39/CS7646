# -*- coding: utf-8 -*-
"""
    @author: achuhan39
    @description : Random Tree Learner for PROJECT-3
    @date : Mar-29-2017

"""
import numpy as np
import scipy.stats as scs

#a=np.array([2,3,54,56,2,3,3,3])
#scs.mode(a)[0][0]

DBG_1 = 0
DBG_2 = 0


class RTLearner(object):

    def __init__(self, verbose = False,leaf_size = 1):
        if(leaf_size < 1) :
            if(verbose): print "Error: Received Leafsize <1. Setting leaf size to 1."
            self.leaf_size = 1
            
        self.leaf_size = round(leaf_size,0)
        #print ("in RT learner - Classification. Leaf =", self.leaf_size )

    def author(self):
        return 'achauhan39 -Classification '
        
    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        #Ytrain_temp = Ytrain[..., None]
        #all_data = np.hstack((Xtrain, Ytrain_temp))
        
        data_xy = np.column_stack((dataX, dataY))
        self.dTree = self.build_tree(data_xy)


      
    def build_tree(self,data) :
    
    	# if num of rows <= leaf size
        if data.shape[0] <= self.leaf_size : 
            #label = np.mean(data[:,-1])
            label = scs.mode(data[:,-1])[0][0]
            return np.array([-1, label, -1, -1]) 
    
        # if all labels are same in remaining rows
        if np.all(data[:,-1] == data[0,-1]): 
            if(DBG_2) :  print "all labels are same"
            return np.array([-1, data[0,-1] , -1, -1])
        
        #if all remaining feature rows have same values
        tmp = data[ : , 0:data.shape[1]-1]
        if np.all(tmp == tmp[0]) : 
            #label = np.mean(data[:,-1])
            label = scs.mode(data[:,-1])[0][0]
            return np.array([-1, label, -1, -1])
            
        else :
            #determine random feature i to split on
            #feature_vec = range((data.shape[1]-1) )
            feature = np.random.randint((data.shape[1]-1))
            if(DBG_2) : print 'Selecting feature :  ' , feature
            
            #SplitVal = (data[random,i] + data[random,i]) / 2
            rand_row = np.random.choice(data.shape[0], size=2 , replace=False)
            if(DBG_1) : print 'random rows : ' ,  rand_row , data[rand_row,feature]
            
            # ***************  handling when feature vals are same 
            if (data[rand_row[0],feature] == data[rand_row[1],feature]) :
                if(DBG_1) : print " WARNING - selected rows have same feature value"
                i=9
                while(i>0) :
                    rand_row = np.random.choice(data.shape[0], size=2 , replace=False)
                    if(DBG_2) : print 'in While loop - random rows : ' ,  rand_row , data[rand_row,feature] 
                    i = i -1
                    if(data[rand_row[0],feature] != data[rand_row[1],feature]) :
                        break
                
            # if data still can't be split after 10 try - select a new feature
            if (data[rand_row[0],feature] == data[rand_row[1],feature]) :
               if(DBG_2) : print " ---  selected rows have same feature value even after 10 try ---"
               if(DBG_2) : print ' -- Selecting NEW feature :  ' , feature
               feature = np.random.randint((data.shape[1]-1))
               
               rand_row = np.random.choice(data.shape[0], size=2 , replace=False)
               if(DBG_2) : print '-- random rows : ' ,  rand_row , data[rand_row,feature]
               
               if (data[rand_row[0],feature] == data[rand_row[1],feature]) :
                   ## Not sure what to do ???
                   #label = np.mean(data[:,-1])
                   label = scs.mode(data[:,-1])[0][0]
                   return np.array([-1, label, -1, -1]) 
                   
            ## *******************************  
                
            #split_val = (data[ rand_row[0], feature] +  data[ rand_row[1], feature])/2
            split_val = (data[rand_row,feature]).mean()
            #split_val = (data[rand_row,feature]).mean()
            
            # build left and right tree recursively
            left_tree  = self.build_tree( data[ data[:,feature] <= split_val] )
            right_tree = self.build_tree( data[ data[:,feature] >  split_val] )
            
            #print "left tree shape and dim = " , left_tree.shape[0] , left_tree.ndim
            
            
            offset = 1 if left_tree.ndim ==1 else left_tree.shape[0] 
            
            root = np.array([feature, split_val, 1, offset + 1])
            #root = [feature, split_val, 1, len(left_tree) + 1]
            
            dtree = np.vstack((root,left_tree, right_tree))            
            return (dtree)
  

  
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result=[]
        for i in range(points.shape[0]):
            lbl = self.get_label(self.dTree , points[i,:])
            result.append(lbl)
        if(DBG_1) :print "len of points : " , points.shape[0]
        if(DBG_1) :print "len of results : " , len(result)
        #print result
        
        return (np.array(result))  
   
   # NonTerminal Node format [ feature ,split_val, left ,right]
   # leaf node format 		 [ -1      ,label    , -1   , -1] 
    def get_label(self,tree, row):
        
        if(tree.shape[0] <=0) :
            #print "Error tree len : " , len(tree)
            return -1
       
        if(tree.ndim==1):
            feature     = int(tree[0])
            split_val   = tree[1]
            left        = int(tree[2])
            right       = int(tree[3])
        else :
            feature     = int(tree[0,0])
            split_val   = tree[0,1]
            left        = int(tree[0,2])
            right       = int(tree[0,3])
        
        # check if its a leaf node
        if (feature == -1) :
           #print " reached leaf node  : label = "  , split_val
           return split_val
       
        # if row[feature] is <= split_val , go left
        if (row[feature] <= split_val) and (left != -1) and (tree[left:, :].shape[0] > 0) :
           return self.get_label(tree[left:, :], row)
        
        # go right   
        if (row[feature] > split_val) and (right != -1) and (tree[right: , :].shape[0] > 0):
           return self.get_label(tree[right: , :], row)
  

if __name__=="__main__":
    print "In RT Learner PROJECT-3"




    
    
    
    
    
