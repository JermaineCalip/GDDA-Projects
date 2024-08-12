import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# To load the dataset into a DataFrame
df = pd.read_csv('Train II.csv', encoding='utf-8-sig')

# To print the first 5 rows for initial observation
print(df.head().to_string())

# To show information about the DataFrame
print('\nInformation about the DataFrame:')
print(df.info())

# Renaming column names
df.columns = df.columns.str.replace('_', ' ').str.replace('.', ' ').str.title()
print('\nColumn Names after renaming:')
print(df.columns)

# To drop unnecessary columns
columns_drop = ['Id']
df = df.drop(columns=columns_drop)
print(f"\nShape after dropping unnecessary columns: {df.shape}")

# Data Conversion
df['Reached On Time Y N'] = df['Reached On Time Y N'].astype(int)

# Data Pre-processing
print('Found in the dataset:')
print(df.isnull().sum())

# To check for duplicated values in the dataset
print('\nDuplicated Values found in the dataset:', df.duplicated().sum())

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
X = df.drop(columns=['Reached On Time Y N'])
y = df['Reached On Time Y N']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Logistic Regression using RandomizedSearchCV
log_reg = LogisticRegression()
param_distributions = {
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}
random_search = RandomizedSearchCV(log_reg, param_distributions, n_iter=100, cv=5,
                scoring='accuracy', random_state=42, return_train_score=True)
random_search.fit(X_train, y_train)

# Finding the best parameters
best_log = random_search.best_estimator_
print(f"\nBest Logistic Regression Parameters: {random_search.best_params_}")
print(f"Best Logistic Regression Score: {random_search.best_score_}")

# Building Regression Models using Training Data
best_log.fit(X_train, y_train)
y_pred = best_log.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy, Precision, Recall, and F1-Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Visualization
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}
plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values(), color='blue')
plt.title('Performance Metrics')
plt.ylabel('Score')
plt.show()

# K-Fold Cross-Validation
cv_scores = cross_val_score(best_log, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy Mean: {cv_scores.mean()}")
print(f"Cross-Validation Accuracy Standard Deviation: {cv_scores.std()}")
