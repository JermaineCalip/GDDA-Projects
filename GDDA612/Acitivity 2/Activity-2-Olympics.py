# To import the necessary packages
import pandas as pd
import seaborn as sns
import numpy as np
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
print('\nNumber of Missing Values:', df.isnull().sum().sum())
# Using the .isnull() and .sum() function, I was able to show the total number of missing values in the table
# Since the result is "0", I don't need to handle missing values

# 2. Remove Duplicates
No_Duplicates = df.duplicated().sum()
print('\nNumber of Duplicated Values:', No_Duplicates)
# I decided to not automatically remove the duplicated values

# To easily remove duplicated records, since the result is '0' I commented out the function
# NoDuplicates = df.drop_duplicates()

# 3. Standardize Data Types
print("\n")
print(df.dtypes)
# To show what type of data for each column, 'object' means strings or non-numeric data types
# Column 'Country' showed object because each country is a String
# The rest of the column is an int64, it implies that it contains numbers

#  4. Address Inconsistent Data
# Capitalization, I want to make all the countries names and header into a title
# For Headers
df.columns = df.columns.str.title()
# For Countries Names
col_name = 'Country' # To define the column name
col_rows = df[col_name].str.title() # to convert string into title
# To print the DataFrame to view the updated column headers and countries names
print(df.head(3).to_string())
print(df.tail(3).to_string())


# 5. Handle Outliers
# To define numerical columns in the data set
numerical_columns = ['Summer', 'Gold', 'Silver', 'Bronze', 'Winter', 'Total',
                     'Combined Games', 'Total Gold', 'Total Silver', 'Total Bronze', 'Combined Total']
# To melt the data set to reshape it into a long format
melteddf = pd.melt(df[numerical_columns], var_name='Column', value_name='Value')

# Create a box plot of the original data
plt.figure(figsize=(12, 6))
sns.boxplot(data=melteddf, x='Column', y='Value')
plt.title('Box Plot of Numeric Columns (Original Data)')
plt.xlabel('Numeric Columns')
plt.ylabel('Values')
plt.show()
# I used trimming to treat outliers
# Define trimming percentage
trim_percentage = 5

# Group melted data by 'Column'
grouped_data = melteddf.groupby('Column')

# Remove outliers based on the trimming thresholds for each column
trimmed_data = pd.DataFrame()
for column, group in grouped_data:
    lower_threshold = np.percentile(group['Value'], trim_percentage / 2)
    upper_threshold = np.percentile(group['Value'], 100 - (trim_percentage / 2))
    trimmed_group = group[(group['Value'] >= lower_threshold) & (group['Value'] <= upper_threshold)]
    trimmed_data = pd.concat([trimmed_data, trimmed_group])

# To create a box plot of the trimmed data
plt.figure(figsize=(12, 6))
sns.boxplot(data=trimmed_data, x='Column', y='Value')
plt.title('Box Plot of Numeric Columns (Trimmed Data)')
plt.xlabel('Numeric Columns')
plt.ylabel('Values')
plt.show()
#NOTE: I got the code for trimming from GeeksforGeeks

# 6. Normalize/Scale Data
numeric_data = df.select_dtypes(include=['int64', 'float64'])
# To calculate the correlation matrix
iris_corr_matrix = numeric_data.corr()
# Print the correlation matrix
print("\nCorrelation Matrix:")
print(iris_corr_matrix)
# To be able to view normalized data through heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(iris_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# To show the title's graph and the graph
plt.title('Correlation Matrix')
plt.show()

# 7.Check and Correct Date Formats
# Since each data types is only objects and integers. There's no need to check and correct date formats

# 8. Text Data Cleaning
# To remove duplicated acronym names enclosed in brackets
df['Country'] = df['Country'].str.replace(r'\s*\[.*?\]\s*', '', regex=True)
print(df.head(3).to_string())
print(df.tail(3).to_string())

# 9. Address Structural Issues
# All column names and unnecessary columns are already addressed on the first part of the code

# 10. Handle Categorical Data
categorical_columns = ['Country']
one_hot_encoded_data = pd.get_dummies(df, columns=categorical_columns, dtype=int)
print(one_hot_encoded_data.to_string())

# 11. Check for Data Integrity
# I addressed all duplicated values, missing values and inconsistent data from the data set

# 12. Create a Data Cleaning Log
with open("Data Cleaning Log.txt", "w") as log_file:
    # To write the header
    log_file.write("Data Cleaning Log for Olympics\n")
    log_file.write("-----------------\n")
    log_file.write("\n")
    # To add description for each header/Process
    log_file.write("All Cleaning Operations Performed\n")
    log_file.write("Deletion: I deleted all the unnecessary characters and words in the data set\n")
    log_file.write("Skip: I skipped unessential rows not needed to be analyze\n")

    log_file.write("Data Cleaning Log for Books\n")
    log_file.write("-----------------\n")
    log_file.write("\n")
    # To add description for each header/Process
    log_file.write("All Cleaning Operations Performed\n")
    log_file.write("Imputation: To addressed missing values. I used imputation rather than deletion because we needed the data\n")
    log_file.write("Deletion: I deleted all the unnecessary characters, dates and columns in the data set\n")

print("Data cleaning log has been created.")

# 13. Validate Cleaned Data
# I selected only the needed columns for Summary Statistics
Specific_Columns = ['Summer', 'Winter', 'Combined Games', 'Total Gold', 'Total Silver', 'Total Bronze', 'Combined Total']
print("\nSummary Statistics for Numerical Variables:")
for var in Specific_Columns:
    print(f"\n{var}:"), # To print column's name
    print(f"  Mean: {df[var].mean()}"), # To print column's mean
    print(f"  Median: {df[var].median()}"), # To print column's median
    print(f"  Standard Deviation: {df[var].std()}"), # To print column's SD
    print(f"  Minimum: {df[var].min()}"), # To print column's min
    print(f"  Maximum: {df[var].max()}"), # To print column's max

print("\nSummary Statistics for Categorical Variables")
for var in df.select_dtypes(include=['object']):
    print(df[var].value_counts())

# 14. Backup the Cleaned Dataset
# To save the cleaned data set to an Excel file
df.to_excel("Cleaned Olympics.xlsx", index=False)













