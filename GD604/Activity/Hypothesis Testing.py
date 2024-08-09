import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('insurance.csv')
print(df.head())

# print(df.isnull().sum())
# print(df.dtypes)

corr = df.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

alpha = 0.5
male_charges = df[df['insuranceclaim'] == 0]['children']
female_charges = df[df['insuranceclaim'] == 1]['children']
t_stat, p_value = stats.ttest_ind(male_charges, female_charges)
print("\nThe t-statistic is: ", t_stat)
print("The p-value is: ", p_value)
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in insuranceclaim between children.")
else:
    print("\nFail to reject the null hypothesis: There is no significant difference in insuranceclaim between children.")

smoker_charges = df[df['smoker'] == 0]['charges']
smokers_charges = df[df['smoker'] == 1]['charges']
t_stat, p_value = stats.ttest_ind(smoker_charges, smokers_charges)
print("\nThe t-statistic is: ", t_stat)
print("The p-value is: ", p_value)
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in smoking between charges.")
else:
    print("\nFail to reject the null hypothesis: There is no significant difference in smoking between charges.")

# Insurance and Age
insurance_age = df[df['insuranceclaim'] == 0]['age']
age_insurance = df[df['insuranceclaim'] == 1]['age']
t_stat, p_value = stats.ttest_ind(insurance_age, age_insurance)
print("\nThe t-statistic is: ", t_stat)
print("The p-value is: ", p_value)
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in insurance claim between age.")
else:
    print("\nFail to reject the null hypothesis: There is no significant difference in insurance claim between age.")

# Insurance and BMI
insurance_bmi = df[df['insuranceclaim'] == 0]['bmi']
bmi_insurance = df[df['insuranceclaim'] == 1]['bmi']
t_stat, p_value = stats.ttest_ind(insurance_bmi, bmi_insurance)
print("\nThe t-statistic is: ", t_stat)
print("The p-value is: ", p_value)
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in insurance claim between bmi.")
else:
    print("\nFail to reject the null hypothesis: There is no significant difference in insurance claim between bmi.")

# CHI square
print('Contingency Table for CHI SQUARE')
contigency_table = pd.crosstab(df['smoker'], df['insuranceclaim'])
chi2, p_value, dof, expected = stats.chi2_contingency(contigency_table)
print(contigency_table)

print(chi2)