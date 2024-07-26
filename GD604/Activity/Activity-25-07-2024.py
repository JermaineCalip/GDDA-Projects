# Import Packages for Analysis
import pandas as pd

# Read Excel File
missing_values = ["n/a", "na", "--", "*", 0]
df = pd.read_csv('data_clean.csv', na_values = missing_values)

# View Data
print(df.to_string())
# Sum of Missing Data
print('\nMissing Values Found in the Dataset')
print(df.isnull().sum())

# 1.Identify and fill or remove missing values in the dataset
# Missing Values in Age Column
avg_age = df['age'].mean()
df['age'] = df['age'].fillna(avg_age)
print('\nAge Column')
print(df['age'])

# Missing Values in Gender Column
df['gender'] = df['gender'].fillna('Male')
print('\nGender Column')
print(df['gender'])

# Missing Values in Email Column after Dropping NaN rows
df = df.dropna(subset=['email'])
print('\nEmail Column')
print(df['email'])

# Missing Values in Join_Date Column
df['join_date'] = df['join_date'].fillna('2020-01-01')
print('\n Join Date Column')
print(df['join_date'])

# Missing Values in Salary Column
df['salary'] = df['salary'].fillna(df['salary'].median())
print('\nSalary Column')
print(df['salary'])

# Missing Values in Department Column
df['department'] = df['department'].fillna('Unknown')
print('\nDepartment Column')
print(df['department'])

# Verify if Missing Values are solved
print('\nDataset After Dealing Missing Values')
print(df.isnull().sum())

# 2. Correct Data Type
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(float)
# For verification
print('\nData Types of Each Column')
print(df.dtypes)

# 3. Standardize Formats
df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce').dt.strftime('%Y-%m-%d')
df['join_date'] = df['join_date'].fillna('1900-01-01')
print('\nStandardized Join Date Column with random Date')
print(df['join_date'])

df['email'] = df['email'].str.lower()
df['email'] = df['email'].str.replace('@@', '@')
df['email'] = df['email'].str.replace(r'\.$', '.com', regex=True)
df['email'] = df['email'].str.replace(r'@com$', '@example.com', regex=True)
print('\nStandardized Email Column')
print(df['email'])

# 4. Remove Duplicates
df.drop_duplicates()
print('\nRemoved Duplicates')
print(df.duplicated())

# 5. Consistent Casing
df['gender'] = df['gender'].str.lower()
print('\nAfter Casing')
print(df.to_string())

# 6. Outlier Detection and Handling
df = df[(df['age'] >= 18) & (df['age'] <= 65)]
print('\nOutlier Detection and Handling')
print(df)

# 7. Correct Invalid Entries
df = df[df['email'].str.contains('@')]
print('\nInvalid Entries in the Email Column')
print(df.to_string())

# 8. Format Column Names
df.columns = df.columns.str.lower().str.replace(' ', '_')
print('\nFormatted Columns Names')
print(df.to_string())

# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
