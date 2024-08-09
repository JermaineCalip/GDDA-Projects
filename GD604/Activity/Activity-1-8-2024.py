import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('employee_data.csv')

# Display the first few rows of the dataset.
print(df.head())

# Display the summary information of the dataset, including data types and non-null counts.
print(df.info())

# Display basic statistical details like mean, median, standard deviation, etc., for numeric columns.
numerical_columns = df.select_dtypes(include=[np.number])
print(numerical_columns.describe())

# Create a bar plot showing the count of employees in each department.
department_counts = df['Department'].value_counts()
department_counts.plot(kind='bar')
plt.title('Number of Employees in Each Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.show()

# Create a histogram of the age distribution of the employees.
plt.figure(figsize=(14, 8))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.show()

# Create a box plot to visualize the salary distribution across different departments.
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Department', y='Salary')
plt.title('Salary Distribution Across Different Departments')
plt.xlabel('Department')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

# Calculate the mean, median, and mode of the salary column.
mean = df['Salary'].mean()
median = df['Salary'].median()
mode = df['Salary'].mode()[0]
print(mean)
print(median)
print(mode)

# Find the department with the highest average salary.
average_salaries = df.groupby('Department')['Salary'].mean()
print(average_salaries.max())

# Calculate the percentage of male and female employees
gender_counts = df['Gender'].value_counts(normalize=True) * 100
print("Percentage of Male Employees:{:2.f}%".format(gender_counts['Male']))
print("Percentage of Male Employees:{:2.f}%".format(gender_counts['Female']))

# Create a correlation matrix for the numeric columns.
corr = numerical_columns.corr()
print(corr)

# Visualize the correlation matrix using a heatmap.
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Create a line plot showing the trend of employee join dates over the years.
df['JoinDate'] = pd.to_datetime(df['JoinDate'])
df['Year'] = df['JoinDate'].dt.year
JoinYear = df['Year'].value_counts().sort_index()
plt.figure(figsize=(14, 8))
plt.plot(JoinYear.index, JoinYear.values, marker='o')
plt.title('Trend of Employee Join Dates Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Employees')
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze if there's any seasonal trend in the hiring of employees.
df['JoinDate'] = df['JoinDate'].dt.month
seasonal_trend = df['JoinDate'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
seasonal_trend.plot(kind='bar')
plt.title('Seasonal Trend in Hiring of Employees')
plt.xlabel('Month')
plt.ylabel('Number of Employees Joined')
plt.grid(True)
plt.show()

# Perform hypothesis testing  on following :
# Gender and Salary
alpha = 0.5
male_salaries = df[df['Gender'] == 'Male']['Salary']
female_salaries = df[df['Gender'] == 'Female']['Salary']
t_stat, p_value = stats.ttest_ind(male_salaries, female_salaries)
print(t_stat)
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in salaries between genders.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in salaries between genders.")

# Department and Salary
departments = df['Department'].unique()
salary_groups = [df[df['Department'] == dept]['Salary'] for dept in departments]
f_stat, p_value = stats.f_oneway(*salary_groups)
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in salaries between departments.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in salaries between departments.")

# Date of joining and Salary


# Build a simple linear regression model to predict salary based on age and department.
# Evaluate the performance of the regression model using R-squared and RMSE.
# Plot the regression line on a scatter plot of age vs. salary