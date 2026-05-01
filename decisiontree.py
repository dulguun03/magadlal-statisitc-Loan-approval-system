# ene codiig ajiluulj uzeegu bga shuu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# 3. Data Cleaning

# Fill missing values
df['loan_amount'] = df['loan_amount'].fillna(df['loan_amount'].mean())
df['loan_term'] = df['loan_term'].fillna(df['loan_term'].mean())

# Fill categorical missing values with most frequent
df['education'] = df['education'].fillna(df['education'].mode()[0])
df['self_employed'] = df['self_employed'].fillna(df['self_employed'].mode()[0])

# Drop unnecessary column
if 'loan_id' in df.columns:
    df.drop('loan_id', axis=1, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# 4. Convert categorical to numeric
le = LabelEncoder()

for col in df.select_dtypes(include='str'):
    df[col] = le.fit_transform(df[col])


print("\nData after encoding:")
print(df.head())


# 5. Split Features and Target
X = df.drop('loan_status', axis=1)
y = df['loan_status']


# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# 7. MODEL 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))


# 8. MODEL 2: Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))


# 9. MODEL 3: Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))


# 10. Compare Models
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_dt)
    ]
})

print("\n=== Model Comparison ===")
print(results)


# 11. Visualization
plt.figure()
plt.bar(results['Model'], results['Accuracy'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.show()


# 12. Example Prediction (NEW DATA)
sample = X.iloc[[0]]

prediction = lr.predict(sample)

print("\nSample Prediction (Logistic Regression):", prediction)

if prediction[0] == 0:
    print("Loan Approved")
else:
    print("Loan Rejected")
