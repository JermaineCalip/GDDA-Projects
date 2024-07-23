import pandas as pd

data ={
    'Name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
    'Age': [24, 30, 22, 35, 29],
    'Score': [85, 90, 95, 80, 88]
}

df = pd.DataFrame(data)

print('Original DataFrame:')
print(df)

# Ascending Order
sorted_df = df.sort_values(by=['Age'])
print('\n DataFrame based on Age:')
print(sorted_df)

# Descending Order
desc_sorted_df = df.sort_values(by=['Score'], ascending=False)
print('\nSorted DataFrame with descending order based on Score:')
print(desc_sorted_df)

data1 = {
    'TransactionID': [1, 2, 3, 4, 5],
    'CustomerID': [101, 102, 103, 104, 105],
    'Amount': [500, 1500, 700, 1200, 300],
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05']
}
df1 = pd.DataFrame(data1)

print('Original DataFrame:')
print(df1)

# Conditional
print('\nDataFrame based on Amount greater than 1000')
sorted_df = df1[df1['Amount'] > 1000]
print(sorted_df)

data2 = {
    'OrderID': [1, 2, 3, 4, 5],
    'Product': ['Apple', 'Banana', 'Orange', 'Grapes', 'Mango'],
    'Quantity': [5, 3, 2, 4, 6],
    'UnitPrice': [2.5, 1.8, 3.0, 2.2, 2.5]
}

df2 = pd.DataFrame(data2)

print('\nOriginal DataFrame:')
print(df2)

df2['TotalPrice'] = df2["Quantity"] * df2["UnitPrice"]
print(df2)

# Group By and Aggregate
data