# -*- coding: utf-8 -*-
"""
    This is 
    Academic Performance Prediction using Machine Learning
"""

from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_excel('grades.xlsx')
df.head()

df.isnull().sum()

# Fill numeric columns with mean
df['StudyHoursPerDay'] = df['StudyHoursPerDay'].fillna(df['StudyHoursPerDay'].mean())
df['SocialMediaHours'] = df['SocialMediaHours'].fillna(df['SocialMediaHours'].mean())
df['SleepHoursPerNight'] = df['SleepHoursPerNight'].fillna(df['SleepHoursPerNight'].mean())
df['Exercise_frequency'] = df['Exercise_frequency'].fillna(df['Exercise_frequency'].mean())
df['mental_health_rating'] = df['mental_health_rating'].fillna(df['mental_health_rating'].mean())
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['MovieTvShowHours'] = df['MovieTvShowHours'].fillna(df['MovieTvShowHours'].mean())
df['AttendancePercentage'] = df['AttendancePercentage'].fillna(df['AttendancePercentage'].mean())
df['Cumulative_Grade'] = df['Cumulative_Grade'].fillna(df['Cumulative_Grade'].mean())

# Fill categorical columns with mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Diet_Quality'] = df['Diet_Quality'].fillna(df['Diet_Quality'].mode()[0])
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])
df['extracurricular_participation'] = df['extracurricular_participation'].fillna(df['extracurricular_participation'].mode()[0])
df['PoR'] = df['PoR'].fillna(df['PoR'].mode()[0])

# Remove rows with missing StudentID (adjust name as needed)

df = df[df['Student_ID'].notna()]

import seaborn as sns
import matplotlib.pyplot as plt

numeric_columns = ['StudyHoursPerDay', 'SocialMediaHours', 'SleepHoursPerNight',
                   'Exercise_frequency', 'mental_health_rating', 'Age',
                   'MovieTvShowHours', 'AttendancePercentage', 'Cumulative_Grade']

for col in numeric_columns:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

df.describe().T[['mean', '50%', 'std']].rename(columns={'50%': 'median'})

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

df['SleepCategory'] = df['SleepHoursPerNight'].apply(lambda x: 'More than 7' if x > 7 else '7 or less')
sns.boxplot(x='SleepCategory', y='Cumulative_Grade', data=df)
plt.title('Grades vs Sleep Duration')
plt.show()

sns.boxplot(x='PoR', y='Cumulative_Grade', data=df)
plt.title('Grades vs Position of Responsibility')
plt.show()

sns.boxplot(x='Diet_Quality', y='Cumulative_Grade', data=df)
plt.title('Grades vs Diet Quality')
plt.show()

sns.scatterplot(x='mental_health_rating', y='Cumulative_Grade', data=df)
plt.title('Mental Health vs Grades')
plt.show()

sns.boxplot(x='extracurricular_participation', y='Cumulative_Grade', data=df)
plt.title('Grades vs Extracurricular Participation')
plt.show()

sns.scatterplot(x='Exercise_frequency', y='Cumulative_Grade', data=df)
plt.title('Exercise vs Grades')
plt.show()

sns.scatterplot(x='AttendancePercentage', y='Cumulative_Grade', data=df)
plt.title('Attendance vs Grades')
plt.show()

df['EntertainmentHours'] = df['MovieTvShowHours'] + df['SocialMediaHours']
sns.scatterplot(x='EntertainmentHours', y='Cumulative_Grade', data=df)
plt.title('TV + Social Media Hours vs Grades')
plt.show()

sns.boxplot(x='parental_education_level', y='Cumulative_Grade', data=df)
plt.title('Grades vs Parental Education Level')
plt.show()

df.to_excel("processed_dataset.xlsx", index=False)
from google.colab import files
files.download("processed_dataset.xlsx")

# Load required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler



#Load the processed dataset

from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_excel('processed_dataset.xlsx')  # Adjust filename if needed
#Separate features and target
y = df['Cumulative_Grade']
X = df.drop(['Cumulative_Grade', 'Student_ID'], axis=1)  # Drop Student_ID along with Cumulative_Grade


#Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling (for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

#Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#Evaluate Models
def evaluate(y_true, y_pred, name):
    print(f"--- {name} ---")
    print("R^2:", r2_score(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("\n")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

#Plot Predictions vs True Labels
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Linear Regression")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Random Forest")
plt.tight_layout()
plt.show()

#Feature Importances
#Linear Regression coefficients
coefs = pd.Series(lr.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("Top 5 Features from Linear Regression:\n", coefs.head())

#Random Forest importances
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 Features from Random Forest:\n", rf_importances.head())

#Error (residual) plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(y_test - y_pred_lr, kde=True)
plt.title("Linear Regression Residuals")
plt.xlabel("Error")

plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_rf, kde=True)
plt.title("Random Forest Residuals")
plt.xlabel("Error")

plt.tight_layout()
plt.show()
