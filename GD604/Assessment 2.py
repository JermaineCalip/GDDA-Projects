# Import necessary packages
import pandas as pd

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
df['order_month'] = df['date'].dt.month
monthly_counts = df.groupby('order_month').size()
print(monthly_counts)

# Aggregate
df_aggregated = df.groupby('nearest_warehouse').agg({'order_total': 'sum'})
print(df_aggregated)









