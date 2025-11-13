import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

df = pd.read_csv("titanic.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

print(df.isnull().sum())

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

print(df.shape)
df.to_csv("cleaned_titanic.csv", index=False)
print("Cleaned dataset saved as cleaned_titanic.csv")
