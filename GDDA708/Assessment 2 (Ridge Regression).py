import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Train I.csv', encoding='ISO-8859-1')
print(df.info())
print("Original shape:", df.shape)

# Dropping unnecessary columns
columns_to_drop = ['Car_id', 'Date', 'Customer Name', 'Phone', 'Dealer_No ']
df = df.drop(columns=columns_to_drop)
print(f"Shape after dropping unnecessary columns: {df.shape}")

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
lower_bound = Q1 - 1 * IQR
upper_bound = Q3 + 1 * IQR

# Remove outliers
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

# Feature (independent variable) and Target (dependent variable)
X = df.drop(columns=['Price ($)'])  # Assuming 'Price ($)' is the target variable
y = df['Price ($)']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Implementing Ridge Regression with GridSearchCV
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Tuning different alpha values
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_ridge = grid_search.best_estimator_
print(f"Best Ridge Parameters: {grid_search.best_params_}")
print(f"Best Ridge Score (negative MSE): {grid_search.best_score_}")

# Predictions and performance metrics
y_pred = best_ridge.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ridge Regression MAE: {mae}")
print(f"Ridge Regression MSE: {mse}")
print(f"Ridge Regression R-squared: {r2}")

