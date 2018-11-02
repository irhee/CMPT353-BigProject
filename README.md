# CMPT353 - Final Project

Stock markets are volatile, but what if we can find a pattern with machine learning? This project uses machine learning techniques and  economic indicators to model a classified S&P 500 index.

The economic data for this project was gathered from Federal Reserve Bank of St.Louis. (https://fred.stlouisfed.org/categories)
The corresponding csv files are in Economy_Indicators_Data folder. We also added Yahoo Finance's S&P500 daily move, and WTO's Import and Export values.

#### Part1_Financial_Data_Parsing_By_Date.py
This program parses all the data gathered using USFederalHolidayCalendar, in order to match the datetime. All of the data is analyzed and moved to four csv files that contains percentage change of the features.
The four csv files are daily, monthly, quarterly, and yearly data. These are saved in Ultimate_Data Folder.

#### Part2_Classification.py
This program analyzes economic data with machine learn techniques such as  Naïve Baysian, K Neighbor, Support Vector Machine, and Principal Component Analysis.
It displays a histogram of the percentage change of the S&P500. We also used PCA to draw two most important features and plotted with Kmean cluster.

Once you have Ultimate Data CSVs ready, simply type-in 'python Part2_Classification.py' on the command line.
