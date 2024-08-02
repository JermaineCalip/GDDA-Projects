# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

# Task 1
# This is to load the dataset into a DataFrame
df = pd.read_csv("Retail Dataset.csv")

# This is to show first 5 rows in the DataFrame
print(df.head().to_string())

# This is to show last 5 rows in the DataFrame
print(df.tail(10).to_string())

# This is to show the total numbers of missing values
print(df.isnull().sum())
# This is to show each column's data type
print(df.dtypes)

# Dealing with Missing Values
# Using Mean Imputation
mean = ['order_price', 'order_total', 'distance_to_nearest_warehouse']
df[mean] = df[mean].fillna(df[mean].mean())

# Dropping Unnecessary Column with Missing Values
drop = ['customer_lat', 'customer_long']
df.drop(drop, axis=1, inplace=True)
df.dropna(subset=['is_happy_customer'], inplace=True)

# Filling NaN with Values
fill = ['season', 'nearest_warehouse']
df[fill] = df[fill].fillna('Unknown')
print(df.isnull().sum())

# Ascending Order
df_asc_sorted = df.sort_values(by='order_total')
print(df_asc_sorted.head().to_string())

# Descending Order
df_desc_sorted = df.sort_values(by='order_total', ascending=False)
print(df_desc_sorted.head().to_string())

# Filtering
df_filtered = df[df['delivery_charges'] > 100]
print(df_filtered.head().to_string())

# New Column
df['date'] = pd.to_datetime(df['date'])
df['order_per_month'] = df['date'].dt.month
monthly_counts = df.groupby('order_per_month').size()
print(monthly_counts)

# Aggregate
df_aggregated = df.groupby('nearest_warehouse').agg({'order_total': 'sum'})
print(df_aggregated)

# Task 2
# Summary Statistics
categorical_columns = df.select_dtypes(include='object')
print(categorical_columns.describe().to_string())

# Correlational
df_encoded = pd.get_dummies(df, columns=['nearest_warehouse', 'season', 'is_happy_customer'])
numerical_columns = df_encoded.select_dtypes(include=['int64', 'float64', 'bool'])
df_corr = numerical_columns.corr()

# Visualization
plt.figure(figsize=(14, 12))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Exporting CSV
df.to_csv('cleaned_retail_dataset.csv', index=False)

# Visualization (Box Plot)
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='season', y='order_total')
plt.title('Total Order per Season')
plt.xlabel('Season')
plt.ylabel('Total Order')
plt.tight_layout()
plt.show()

# Visualization (Bar Plot)
sales = df.groupby('nearest_warehouse')['order_total'].sum().reset_index()
sorted = sales.sort_values(by='order_total', ascending=False)
plt.figure(figsize=(14, 8))
sns.barplot(data=sorted, x='nearest_warehouse', y='order_total')
plt.title('Total Order per Nearest Warehouse')
plt.xlabel('Nearest Warehouse')
plt.ylabel('Order Total')
plt.tight_layout()
plt.show()

# Inferential Analysis

t_stat, p_val = ttest_ind(df['order_total'], df['order_price'])
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
if p_val < alpha:
    print("Reject the null hypothesis: There is a significant difference between order_total and order_price.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between order_total and order_price.")


