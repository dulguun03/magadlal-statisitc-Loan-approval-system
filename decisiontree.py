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

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# 3. Data Cleaning

# Fill missing values
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(1, inplace=True)

# Fill categorical missing values with most frequent
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

# Drop unnecessary column
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# 4. Convert categorical to numeric
le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])


print("\nData after encoding:")
print(df.head())


# 5. Split Features and Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']


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
sample = X.iloc[0].values.reshape(1, -1)

prediction = lr.predict(sample)

print("\nSample Prediction (Logistic Regression):", prediction)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
