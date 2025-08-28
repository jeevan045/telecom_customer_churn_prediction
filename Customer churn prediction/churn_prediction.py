# Churn Prediction and Analysis - Complete Fixed and Working Code

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer

# 1. Load the dataset (replace with your file path)
df = pd.read_csv('customer_churn_data.csv')

# 2. Data Exploration
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
print("\nOriginal Churn values:", df['Churn'].head(20))
print("Unique original Churn values:", df['Churn'].unique())

# 3. Data Preprocessing
# Handle missing/empty values in TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
imputer = SimpleImputer(strategy='mean')
df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

# Fix Churn column: fill NaN and encode
df['Churn'] = df['Churn'].fillna('No')  # Replace NaN with 'No'
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Churn':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# 4. Feature Engineering: Create tenure_group and encode it
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=['0-12', '12-24', '24-48', '48-60', '60+'])
# Encode tenure_group as a categorical variable
le = LabelEncoder()
df['tenure_group'] = le.fit_transform(df['tenure_group'].astype(str))

# 5. Data Visualization
print("After encoding, unique Churn values:", df['Churn'].unique())

plt.figure(figsize=(12, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn')
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.show()

# 6. Prepare data for modeling
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 7. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Model Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# 9. Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# 10. Save results for Power BI
results = X_test.copy()
results['Actual'] = y_test
results['Predicted'] = y_pred
results['Probability'] = y_proba
results.to_csv('churn_predictions.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("\nPredictions and feature importance saved to CSV files for Power BI integration.")
