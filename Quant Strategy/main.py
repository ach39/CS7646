# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 21:54:00 2017

File name : main.py

@author: achauhan39
Version -Final( Apr-09)

"""
from Indicators import plot_normalised_Indicators
from rule_based import Q6,best_strategy,run_manual_trader
from ML_based import run_ML_trader,init


#################################################
# Valid range of options is 1...5
#   select option
#       0 to run all options from 1 to 5 in one go
#       1 to plot indicators
#       2 to execute best strategy
#       3 to run manual trader
#       4 to run ML trader
#       5 to generate scatter plot (SSO vs BB%) of ML in-sample
#
 #################################################

instructions = "\
   select option \n \
       #0 to run all options from 1 to 5 \
       #1 to plot indicators. \n \
       #2 to execute best strategy. \n \
       #3 to run manual trader. \n \
       #4 to run ML trader. \n \
       #5 to generate scatter plot (SSO vs BB%) of ML in-sample data."
      
option = 4

if option==1 :
    plot_normalised_Indicators()

elif option==2:
    best_strategy()

elif option==3:
    run_manual_trader(version=1,out=False)
    run_manual_trader(version=2,out=False)

elif option==4:
    run_ML_trader(scatter=False,out=False)

elif option==5:
    run_manual_trader(scatter=True)
    run_ML_trader(scatter=True)

#elif option ==6:
#    Q6()
    
elif option==0 :
    plot_normalised_Indicators()
    best_strategy()
    run_manual_trader(version=1,scatter=True,out=False)
    run_ML_trader(scatter=True,out=False)
   
   
else:
    print "Invalid option : " , option
    print  instructions
     
