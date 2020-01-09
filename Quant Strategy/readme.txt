
Goal: Develop trading strategies using Technical Analysis, and test them using your market simulator. You will then utilize your Random Tree learner to train and test a learning trading algorithm. 

-------------------------------------------------------------
Data Details, Dates and Rules
-------------------------------------------------------------
 1. Use only the data provided for this course. You are not allowed to import external data.
 2. Trade only the symbol AAPL (however, you may, if you like, use data from other symbols to inform your strategy).
 3. The in sample/training period is January 1, 2008 to December 31 2009.
 4. The out of sample/testing period is January 1, 2010 to December 31 2011.
 5. Starting cash is $100,000.
 6. Allowable positions are: 200 shares long, 200 shares short, 0 shares.
 7. Benchmark: The performance of a portfolio starting with $100,000 cash, investing in 200 shares of AAPL and holding that position
 8. There is no limit on leverage.

-------------------------------------------------------------
How to reproduce results ?
-------------------------------------------------------------

Driver code for mc3p3 is implemneted in main.py, which allows user to select an option to view desired output


set 'option' variable in main.py to one of the following values 
	
       0 to run all options 1 to 5 in one go
       1 to plot indicators
       2 to execute best strategy
       3 to run manual trader
       4 to run ML trader
       5 to generate scatter plot (SSO vs BB%) of ML in-sample data.


option=0
	This will execute all the options mentioned below in one go.

option=1
	When option is set to 1, code generates graphs for techincal indicators used for this project.

option=2
	This option generates 
	- an orders file called 'best_strategy_orders.csv' 
	- a graph comparing best-strategy portfolio vs. benchmark

option=3
	This option runs manual trader for in-sample period and generates 
	- an orders file called 'rule_based_orders.csv' 
	- a graph comparing manual rule-based portfolio vs. benchmark

option=4
	This option runs ML trader for in-sample period and generates 
	- an orders file called 'ML_orders_train.csv' 
	- a graph comparing ML-based portfolio vs. benchmark

option=5 
	This option is same as option 3 and 4 except it also generates 
	- scatter plot (SSO vs BB%) as required in part-5 of the assignment



###############################################################
Project Files
###############################################################

New files
	main.py
	Indicators.py
	ML_based.py
	rule_based.py

Refactored code from previous projects
	marketsim.py
	RTLearner.py
	BagLearner.py
	util.py


