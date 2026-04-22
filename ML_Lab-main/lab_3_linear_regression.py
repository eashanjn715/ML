import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Part 1: Per Capita Income
df_canada = pd.read_csv('canada_per_capita_income.csv')
X = df_canada[['year']]
y = df_canada['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)
prediction_2020 = model.predict([[2020]])
print(f"Predicted Per Capita Income for 2020: ${prediction_2020[0]:.2f}")

# Part 2: Salary
df_salary = pd.read_csv('salary.csv')
df_salary['YearsExperience'] = df_salary['YearsExperience'].fillna(df_salary['YearsExperience'].median())
X_salary = df_salary[['YearsExperience']]
y_salary = df_salary['Salary']

model_salary = LinearRegression()
model_salary.fit(X_salary, y_salary)
predicted_salary = model_salary.predict([[12]])
print(f"Predicted Salary for 12 years of experience: ${predicted_salary[0]:.2f}")
