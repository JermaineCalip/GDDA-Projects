import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# To load the dataset into a DataFrame
df = pd.read_csv('Train I.csv', encoding='ISO-8859-1')

# To print the first 5 rows for initial observation
print(df.head().to_string())

# To show information about the DataFrame
print('\nInformation about the DataFrame:')
print(df.info())

# To drop unnecessary columns
columns_drop = ['Car_id', 'Date', 'Customer Name', 'Phone', 'Dealer_No ']
df = df.drop(columns=columns_drop)
print(f"\nShape after dropping unnecessary columns: {df.shape}")

# Data Pre-processing
# To check for missing values in the dataset
print('\nMissing Values found in the dataset:')
print(df.isnull().sum())

# To check for duplicated values in the dataset
print('\nDuplicated Values found in the dataset:', df.duplicated().sum())
df = df.drop_duplicates()
# To verify
print('Total Numbers of Duplicates after dropping:', df.duplicated().sum())

# To separate numerical and categorical columns
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
lower_bound = Q1 - 1 * IQR
upper_bound = Q3 + 1 * IQR

# To remove outliers
outliers = ((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound))
df = df[~outliers.any(axis=1)]

# For loop for visualization for each column
for col in numerical_columns:
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df, x=col)
    plt.title(f'Box Plot of {col} after Removing Outliers')
    plt.show()

# One-hot Encoding
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
print(f"\nFinal shape after one-hot encoding: {df.shape}")

# Selecting Feature (independent variable) and Target (dependent variable)
X = df.drop(columns=['Price ($)'])
y = df['Price ($)']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Ridge Regression using RandomizedSearchCV
ridge = Ridge()
param_distributions = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
}
random_search = RandomizedSearchCV(ridge, param_distributions, n_iter=50, cv=5,
                scoring='neg_mean_squared_error', random_state=42,)
random_search.fit(X_train, y_train)

# Finding the best parameters
best_ridge = random_search.best_estimator_
print(f"\nBest Ridge Parameters: {random_search.best_params_}")
print(f"Best Ridge Score: {random_search.best_score_}")

# Building Regression Models using Training Data
best_ridge.fit(X_train, y_train)
y_pred = best_ridge.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nRidge Regression MAE: {mae}")
print(f"Ridge Regression MSE: {mse}")
print(f"Ridge Regression R-squared: {r2}")

# K-Fold Cross-Validation
cross = cross_val_score(best_ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"\nCross-Validation MSE Mean: {cross.mean()}")
print(f"Cross-Validation MSE Standard Deviation: {cross.std()}")