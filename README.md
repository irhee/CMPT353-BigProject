# CMPT353-BigProject

Stock markets are volatile; but, what if we can find a pattern with machine learning? We used machine learning techniques and fed economic indicators to model a classified S&P 500 index.

We gathered economic data from Federal Reserve Bank of St.Louis. (https://fred.stlouisfed.org/categories)
Those csv files are in Economy_Indicators_Data folder. We also added Yahoo finance S&P500 dilay move, and WTO's Import and Export values.

Part1_Financial_Data_Parsing_By_Date.py
Parses all the data gathered using USFederalHolidayCalendar, in order to match the datetime. The program assorts all the data and creates four csv files that contains percentage change of the features.
The four csv files are daily, monthly, quarterly, and yearly data. These are saved in Ultimate_Data Folder.

Part2_Classification.py
This program analyzes the economy data with machine learn technique such as  Naïve Baysian, K Neighbor, Support Vector Machine, and Principal Component Analysis.
It displays the histogram of the percentage change of the S&P500. We also used PCA to draw two most important features and plotted with Kmean cluster.

Once you have Ultimate Data CSVs ready, simply type-in 'python Part2_Classification.py' on the command line.