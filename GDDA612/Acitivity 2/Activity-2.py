# Import the necessary packages
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# To read the csv file and pass it to the variable
DBOlympics = pd.read_csv('olympics.csv')

# I used to_string() function to show the whole table (Pycharm)
print(DBOlympics.to_string())

# 1. Identify and Handle Missing Values
# To show the total number of missing values in the table
# I used iloc function to not consider the first row because I wanted to filter only the data, not the header
print('Number of Missing Values:', DBOlympics.iloc[1:].isnull().sum().sum())
# Since the result is "0", I don't need to handle missing values

# 2. Remove Duplicates
# I decided to not remove duplicated values instantly because maybe some values have same output/values normally
# Count the number of duplicated records
NoDuplicates = DBOlympics.duplicated().sum()

# To print the number of duplicated rows
print('Number of duplicated records:', NoDuplicates)

# To remove duplicated records, since the result is '0' I commented out the function
# df = DBOlympics.drop_duplicates()

#  3. Standardize Data Types
# To show what type of data each column, 'object' means strings, integers, floats or non-numeric data types
# All columns showed 'object', it implies that there is no date formatted columns, there's no need to standardize data types
print('Data types of each column:')
print(DBOlympics.dtypes)

# Code snippet to change selected column to formatted date using datetime
# DBOlympics['date_column'] = pd.to_datetime(DBOlympics['date_column'])
# DBOlympics['numerical_column'] = DBOlympics['numerical_column'].astype(float)

#  4. Address Inconsistent Data
# I need to change 'NaN' to 'Country' for consistency
DBOlympics.fillna('Country', inplace=True)
# Just to print the top 3 for smaller view of the table
# print(DBOlympics.head(3).to_string())

# 5. Handle Outliers
# z_scores = (DBOlympics - DBOlympics.mean()) / DBOlympics.std()
# outliers = (z_scores > 3) | (z_scores < -3)

# 6. Normalize/Scale Data

# 7.Check and Correct Date Formats
# There's no need to check and correct date formats

# 8. Text Data Cleaning
# I removed the first row of the data set because it's not important and makes renaming easier
# I must remove all the necessary characters and punctuation marks
import pandas as pd

# Assuming DBOlympics is your DataFrame
# Convert the first column to numeric, coercing errors
DBOlympics['Country'] = pd.to_numeric(DBOlympics['Country'], errors='coerce')

# Remove rows where the first column contains numeric values
DBOlympics = DBOlympics[DBOlympics['Country'].isnull()]

# Reset index
DBOlympics.reset_index(drop=True, inplace=True)

# Print the DataFrame to verify the changes
print(DBOlympics.head())

# DBOlympics.rename(columns = {"0":"Percentage"}, inplace=True)
# print(DBOlympics.head(3).to_string())







