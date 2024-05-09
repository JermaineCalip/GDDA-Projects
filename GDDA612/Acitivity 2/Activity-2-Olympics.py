# To import the necessary packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# To read the Excel file and convert it to a DataFrame
# I used skiprows=1 to skip the first row because it's unnecessary
df = pd.read_csv('olympics.csv', skiprows=1)

# First, it is better to organize first the header of a data set to familiarize each column connection
# And, so as not to mess up each column
df.rename(columns={'Unnamed: 0': 'Country', '? Summer': 'Summer', '01 !': 'Gold', '02 !': 'Silver', '03 !': 'Bronze',
                   '? Winter': 'Winter', '01 !.1': 'Gold', '02 !.1': 'Silver', '03 !.1': 'Bronze', 'Total.1': 'Total',
                   '? Games': 'Combined Games', '01 !.2': 'Total Gold', '02 !.2': 'Total Silver', '03 !.2': 'Total Bronze',
                   }, inplace=True)
# To rename each column of a data set at once
# I changed the columns name to '1 = Gold', '2 = Silver', 3 = 'Bronze' because the data set is about the olympics which
# every countries try to compete, represent and earn medals.

# To print only the top and bottom 3 rows to easily view the data
print(df.head(3).to_string())
print(df.tail(3).to_string())

# START OF THE ACTIVITY #

# 1. Identify and Handle Missing Values
print('Number of Missing Values:', df.isnull().sum().sum())
# Using the .isnull() and .sum() function, I was able to show the total number of missing values in the table
# Since the result is "0", I don't need to handle missing values

# 2. Remove Duplicates
No_Duplicates = df.duplicated().sum()
print('Number of Duplicated Values', No_Duplicates)
# I decided to not automatically remove the duplicated values

# To easily remove duplicated records, since the result is '0' I commented out the function
# NoDuplicates = df.drop_duplicates()

# 3. Standardize Data Types
print(df.dtypes)
# To show what type of data for each column, 'object' means strings or non-numeric data types
# Column 'Country' showed object because each country is a String
# The rest of the column is an int64, it implies that it contains numbers

#  4. Address Inconsistent Data
# Capitalization, I want to make all the countries names and header into a title
# For Headers
df.columns = df.columns.str.title()
# For Countries Names
colname = 'Country' # To define the column name
colrows = df[colname].str.title() # to convert string into title
# To print the DataFrame to view the updated column headers and countries names
print(df.head(3).to_string())
print(df.tail(3).to_string())

# 5. Handle Outliers
sns.boxenplot(x=df['Combined Games'])
plt.title('Boxen plot of Numeric Column with Outliers')
plt.show()

# 6. Normalize/Scale Data

# 7.Check and Correct Date Formats
# Since each data types is only object and integers. There's no need to check and correct date formats

# 8. Text Data Cleaning


# 9. Address Structural Issues












