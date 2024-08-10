import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Train II.csv', encoding='ISO-8859-1')
print(df.info())
print("Original shape:", df.shape)
print(df.columns)

# # Dropping unnecessary columns
# columns_to_drop = ['Car_id', 'Date', 'Customer Name', 'Phone', 'Dealer_No ']
# df = df.drop(columns=columns_to_drop)
# print(f"Shape after dropping unnecessary columns: {df.shape}")

# Data Pre-processing
# Check for missing values
print('\nMissing Values found in the dataset:')
print(df.isnull().sum())

# Check for duplicated values
print('\nDuplicated Values found in the dataset:', df.duplicated().sum())

# Separating numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Standardization
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Calculate IQR for outlier detection
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Outlier Boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
outliers = ((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound))
df = df[~outliers.any(axis=1)]

# For loop for visualization for each column
for col in numerical_columns:
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df, x=col)
    plt.title(f'Box Plot of {col} after Removing Outliers')
    plt.show()



