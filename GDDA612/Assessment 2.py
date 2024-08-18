# Importing Necessary Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.parquet as pq
from pymongo import MongoClient, ASCENDING

# This is to load the dataset into a DataFrame
df = pd.read_csv('E-commerce.csv', encoding='ISO-8859-1')
# Load and Analyze the structure of Dataset
print(df.head().to_string(),'\n')
print(df.info())

# Pre-processing
# To find missing values in the dataset
print('\nMissing Values found in the dataset:')
print(df.isnull().sum())

# Distribution and Skewness
# missing_columns = ['Columns with missing Values']
# for col in missing_columns:
#     plt.figure(figsize=(10, 5))
#     sns.histplot(df[col], kde=True)
#     plt.title(f'Histogram of {col}')
#     plt.show()

# To fill missing values in the dataset
df['Description'] = df['Description'].fillna('Unknown')
df = df.drop(columns=['CustomerID'])

# To find duplicated values in the dataset
print("\nTotal No. of Duplicates:", df.duplicated().sum())

# To verify if missing values still present in the dataset
print("\nTotal No. of Missing Values after:")
print(df.isnull().sum())

# To detect outlier detection using IQR
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Calculate IQR
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Outlier Boundaries
lower_bound = Q1 - 0.4 * IQR
upper_bound = Q3 + 0.4 * IQR

# To detect outliers
outliers = ((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound))

# To remove outliers
df = df[~outliers.any(axis=1)]

# Summary after removing outliers
total_outliers = outliers.sum().sum()
after_removal = df.shape[0]

# To print the total and shape of the dataset after removing outliers
print(f"\nTotal number of outliers: {total_outliers}")
print(f"Total number of rows after outlier removal: {after_removal}")

# For loop for visualization for each column
for col in numerical_columns:
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df, x=col)
    plt.title(f'Box Plot of {col} after Removing Outliers')
    plt.show()

# Data Type Conversion
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
df['InvoiceDate'] = df['InvoiceDate'].dt.date
df['InvoiceDate'] = df['InvoiceDate'].astype(str)

# To verify
print("\nData Types After Conversion:")
print(df.dtypes)

# To show tidy dataset
print('\nDataset after Data Cleaning and Preprocessing:')
print(df.head().to_string())
print(df.shape)

# Filtering based on Specified Criteria
print('\nRetrieved Dataset based on UnitPrice and Country')
filtered_df = df[(df['UnitPrice'] == 2) & (df['Country'] == 'United Kingdom')]
print(filtered_df.to_string())

# Establishing Connection to MongoDB
data_dict = df.to_dict(orient='records')
client = MongoClient('mongodb+srv://jermainejancalip27:Jermaine27.@cluster0.vuirs2r.mongodb.net/')
# Check the connection if successful
print("\nConnection established successfully!")

# Create and Insert Data into MongoDB
db = client['E-Commerce']
collection = db['E-Commerce']
# collection.insert_many(data_dict)

# Retrieve and display one document from MongoDB
# document = collection.find()
document = collection.find_one()
print('\nDisplaying one document from MongoDB:', document)

# Sorting based on Specified Criteria
print('\nSorted with specified criteria')
sorted_df = collection.find({'Country': 'France'}).sort('Quantity', ASCENDING).limit(5)
# To print each document
for document in sorted_df:
    print(document)

# Count the total number of records
total_documents = collection.count_documents({})
print('\nTotal number of documents:', total_documents)

# Grouping Operation
documents = list(collection.find())
mongo_df = pd.DataFrame(documents)
# Using Group By and Aggregation
grouped_df = mongo_df.groupby('Country')['Quantity'].sum().reset_index()
print(grouped_df)

# Update Operation
collection.update_many({'Country': 'United Kingdom'}, {'$set': {'Country': 'UK'}})
# To print each document
for doc in collection.find({'Country': 'UK'}).limit(5):
    print(doc)

# Parquet File
cursor = collection.find({})
cleaned_df = pd.DataFrame(list(cursor))

# Dropping the '_id' because of conversion error
cleaned_df = cleaned_df.drop(columns=['_id'])

# Convert DataFrame to Parquet
table = pa.Table.from_pandas(cleaned_df)
pq.write_table(table, 'E-Commerce')

# Close the connection
client.close()

