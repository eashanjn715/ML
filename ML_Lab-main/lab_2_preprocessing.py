import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

data = {
    'Age': [25, 30, 35, np.nan, 200, 22, 38, 45, 12, np.nan],
    'Salary': [50000, 54000, np.nan, 62000, 58000, 52000, 120000, 60000, 48000, 55000],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT', np.nan, 'Sales', 'IT', 'HR'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])

le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Age'] = np.where(df['Age'] > upper_bound, upper_bound,
                     np.where(df['Age'] < lower_bound, lower_bound, df['Age']))
