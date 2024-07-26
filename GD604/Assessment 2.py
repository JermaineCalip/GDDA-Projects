# Import necessary packages
import pandas as pd

# This is to load the dataset into a DataFrame
df = pd.read_csv("Sales_Sample_Public_Dataset.csv", encoding="ISO-8859-1")

# This is to show few rows in the DataFrame
print(df.head(5).to_string())

# This is to show the total numbers of missing values
print(df.isnull().sum())

# # This is to solve the missing values found in categorical columns using mode imputation.
# categorical_columns = df.select_dtypes(include=['object']).columns
# df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
# # This is to display missing values after mode imputation for verify.
# # print(df.isnull().sum(),"\n")
#
# # This is to solve the missing values found in numerical columns using mean imputation.
# numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
# df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
# # This is to display missing values after mean imputation for verify.
# # print(df.isnull().sum(),"\n")
#
# print(df.isnull().sum().sum())
#
# df.to_csv('test.csv', index=False)
#
# df = pd.read_csv('test.csv')
# print (df.head(5).to_string())
#
# print(df.dtypes)

# sns.boxplot(df.)
# sns.distplot(df.salary)
