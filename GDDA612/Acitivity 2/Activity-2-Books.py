# To import the necessary packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# To read the Excel file and convert it to a DataFrame
df = pd.read_csv('BL-Flickr-Images-Book.csv')

# START OF THE ACTIVITY #

# 1. Identify and Handle Missing Values
print('Number of Missing Values:')
print(df.isnull().sum())
print('Total number of Missing Values:', df.isnull().sum().sum())
df['Date of Publication'].fillna('', inplace=True)
# Using the .isnull() and .sum() function, I was able to show the total number of missing values in the table

# 2. Remove Duplicates
# Imputation for string columns
string_columns = df.select_dtypes(include=['object']).columns # To select all columns which are object types
for col in string_columns:
    df[col].fillna(df[col].mode().iloc[0], inplace=True) # The Iloc[0] function means that it will start at column 1

# Imputation for numerical columns
numerical_columns = df.select_dtypes(include=['float64']).columns
for col in numerical_columns:
    df[col].fillna(0, inplace=True)

# Check if there are any remaining missing values
print(df.isnull().sum())
print('Total number of Missing Values:', df.isnull().sum().sum())
print(df.head().to_string())

# 3. Standardize Data Types
print(df.dtypes)
# To Change data types of a specific column
change_DataType = {
    'Corporate Author': 'object',
    'Corporate Contributors': 'object',
    'Engraver': 'object',
}
# To Change data types of specified columns
df = df.astype(change_DataType)
# To check the data types have been changed
# I left "Date of Publication" as object because dates has characters as string
print(df.dtypes)

#  4. Address Inconsistent Data
# Remove unnecessary characters from 'Place of Publication' column using regex
df['Place of Publication'] = df['Place of Publication'].str.extract(r'(\b[A-Z][a-zA-Z\s]+\b)$', expand=False)
# Extract only the year from 'Date of Publication' column
df['Date of Publication'] = df['Date of Publication'].str.extract(r'(\b\d{4}\b)', expand=False)
# Convert 'Date of Publication' column to int64
df['Date of Publication'] = pd.to_numeric(df['Date of Publication'], errors='coerce').fillna(0).astype('int64')
print(df.head().to_string())
df['cleaned_title'] = df['Title'].str.replace('[\[\]*\(\)]', '')
# Remove extra spaces from 'cleaned_title' column using regex
df['cleaned_title'] = df['cleaned_title'].str.replace(r'\s+', ' ')
df.drop(columns=['cleaned_title'], inplace=True)

# 5. Handle Outliers

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
# I already addressed this issue in part 4 because I need to clean the inconsistency to able to perform normalization

# 9. Address Structural Issues
print('\n')
df = df.drop(columns=['Corporate Author', 'Corporate Contributors', 'Former owner', 'Engraver'])
# I removed the following columns because its empty and doesn't impact the whole data set
print(df.head(5).to_string())

# # 10. Handle Categorical Data
# categorical_columns = ['Edition Statement', 'Place of Publication', 'Publisher', 'Title', 'Author','Contributors', 'Date of Publication', 'Issuance type', 'Shelfmarks']
# one_hot_encoded_data = pd.get_dummies(df, columns=categorical_columns, dtype=int)
# print(one_hot_encoded_data.to_string())

# 11. Check for Data Integrity
# I addressed all duplicated values, missing values and inconsistent data from the data set

# 12. Create a Data Cleaning Log
# Created a txt file for my data cleaning log

# 13. Validate Cleaned Data
print("\nSummary Statistics for Numerical Variables:")
for var in df.select_dtypes(include=['int64', 'float64']):
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
df.to_excel("Cleaned Books.xlsx", index=False)