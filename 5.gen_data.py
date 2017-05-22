"""
template for generating data to fool learners (c) 2016 Tucker Balch
@author - achauhan@39
version - Mar-18 :  add random seed generator 
Mar-25 : Passed autograder
"""

import numpy as np
import matplotlib.pyplot as plt

def author():
        return 'achauhan39'

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=5):
    
    np.random.seed(seed)
    #ROW_CNT = np.random.randint(850, 950)
    ROW_CNT = 950
     
    X = np.random.normal(5,10,size = (ROW_CNT, 2)) 
    noise = np.random.standard_exponential(ROW_CNT)    
    #Y = 6 + np.pi *X[:,0] + 3* X[:,1] + noise
    Y = 6.0 + np.pi *X[:,0] + 3.0 * X[:,1] 
    
    #Y = np.sin(X[:,1])*np.cos(1./(0.0001+X[:,0]**2)) 
    
#    plt.scatter(X[:,0], Y)
#    plt.scatter(X[:,1], Y,c='r')
#    plt.show()
    
    return X, Y



def best4RT(seed=5):
    np.random.seed(seed)
    center = [-5, 10 , 0]
    std = 2.0
    features = 3
    label = np.random.choice(range(-20,20), size=3 , replace=False)
    #label = [1,2,3]
    data=np.zeros(shape=(900,features+1))
    
    for row in range(0,300):
      data[row,:3] = [center[0] + np.random.random() * std for i in range(features)]
      data[row,-1] = label[0]
    
    for row in range(300,600):
      data[row,:3] = [center[1] + np.random.random() * std for i in range(features)]
      data[row,-1] = label[1]
      
    for row in range(600,900):
      data[row,:3] = [center[2] + np.random.random() * std for i in range(features)]
      data[row,-1] = label[2]
    
    X = data[:, :-1]
    Y = data[:, -1]  
      
#    plt.scatter(X[:,0], Y)
#    plt.scatter(X[:,1], Y,c='r')
#    plt.show()
    
    return X,Y   
    
    

if __name__=="__main__":
    print author()
    
#    X,Y = best4LinReg(seed=400)
#    X1,Y1 = best4LinReg(seed=20)
    
#    X,Y = best4RT(seed=30)
#    X1,Y1 = best4RT(seed=20)
#    
#    print X1.shape ,  Y1.shape
#    print X.shape ,  Y.shape  
#    print np.array_equal(X,X1) , np.array_equal(Y,Y1)

    
    






