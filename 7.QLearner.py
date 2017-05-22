"""
Template for implementing QLearner  (c) 2015 Tucker Balch
@achauhan39
@date - Apr-21
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0
        #self.Q = np.zeros((self.num_states, self.num_actions))
        self.Q = np.random.uniform(-1.0,1.0, size=(self.num_states ,self.num_actions))
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states ))    # T[s, a ,s']
        self.Tc = np.full((self.num_states, self.num_actions, self.num_states), 0.00001)    
        self.R = np.zeros((self.num_states, self.num_actions))                      # R[s', a]
        self.exp_list=[] 

    def author(self):
        return 'achauhan39'
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s          
              
        # select action
        p = rand.uniform(0.0,1.0)
        if(p < self.rar) :
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s])

        if self.verbose: print "s =", s,"a =",action
        
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.exp_list.append([self.s, self.a ,s_prime , r])
        self.update_Q(self.s, self.a, s_prime, r)
        
        #self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * max(self.Q[s_prime]))
        
                            
#        #update Tc
#        self.Tc[self.s, self.a,s_prime] += 1
#        
#        #update Model
#        self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + self.alpha * r
#        self.T[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] / self.Tc[self.s, self.a,:].sum()
        
        self.run_DynaQ(s_prime, r)
        
        # select action
        p = rand.uniform(0.0,1.0)
        if(p < self.rar) :
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s_prime])
        
        #update rar
        self.rar  *= self.radr
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
         
        #update local s and a
        self.s = s_prime
        self.a = action
        
        return action


    def update_Q(self,s, a, s_prime, r) :
        self.Q[s, a] = (1-self.alpha) * self.Q[s, a] + \
                                    self.alpha * (r + self.gamma * max(self.Q[s_prime]))
        
        #f_reward = self.Q[s_prime, np.argmax(self.Q[s_prime])]
        #self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * f_reward)
        

    
    
    def run_DynaQ(self,s_prime,r):
            
        # hallucinate
        exp_cnt = len(self.exp_list)
        idx = np.random.randint(0,exp_cnt, self.dyna)
        
        for i in idx :
            exp_tup = self.exp_list[i]
            s = exp_tup[0]
            a = exp_tup[1]
            #s_prime = self.T[s,a,s_prime]
            #r = self.R[s,a]
            s_prime = exp_tup[2]
            r = exp_tup[3]

            self.update_Q(s,a, s_prime, r)
            
        
        


if __name__=="__main__":
    print "achauhan39"

            
