import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- 1. LOAD DATA ---
print("Loading data...")
df = pd.read_csv('BankChurners.csv')

# --- 2. CLEANING ---
cols_to_drop = [
    'CLIENTNUM', 
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
]
df = df.drop(columns=cols_to_drop, errors='ignore')

# --- 3. FIX TARGET VARIABLE ---
df['Attrition_Flag'] = df['Attrition_Flag'].astype(str).str.strip()
target_map = {'Existing Customer': 0, 'Attrited Customer': 1}
df['Attrition_Flag'] = df['Attrition_Flag'].map(target_map)
df = df.dropna(subset=['Attrition_Flag'])
df['Attrition_Flag'] = df['Attrition_Flag'].astype(int)

# --- 4. ENCODE FEATURES ---
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# --- 5. SPLIT & BALANCE (SMOTE) ---
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Data balanced successfully.")

# --- 6. TRAIN THE MODEL (Random Forest) ---
print("Training the Random Forest model... (This may take 10-20 seconds)")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# --- 7. EVALUATE ---
print("\nPredicting on Test Data...")
y_pred = rf_model.predict(X_test)

print("\n--- CONFUSION MATRIX ---")
# [True Neg  False Pos]
# [False Neg True Pos]
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# --- 8. SHOW WHAT MATTERS MOST ---
# This shows which features indicate fraud/churn the most
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("\nTop 5 Most Important Features:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]}")


# 1. Pick a random row from the Test Data (Data the model hasn't seen for training)
random_index = random.choice(X_test.index)
customer_data = X_test.loc[[random_index]]
actual_status = y_test.loc[random_index]

# 2. Ask the model to predict
prediction = rf_model.predict(customer_data)
probability = rf_model.predict_proba(customer_data)

# 3. Show the results
print("\n--- SINGLE CUSTOMER TEST ---")
print(f"Customer Details (Encoded):\n{customer_data.values}")
print(f"\nACTUAL Status:    {'Attrited (Fraud/Left)' if actual_status == 1 else 'Existing (Safe)'}")
print(f"MODEL Prediction: {'Attrited (Fraud/Left)' if prediction[0] == 1 else 'Existing (Safe)'}")
print(f"Confidence Score: {probability[0][prediction[0]] * 100:.2f}%")

if actual_status == prediction[0]:
    print("✅ RESULT: CORRECT")
else:
    print("❌ RESULT: WRONG")

# Create the plot
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', display_labels=["Existing", "Attrited"])
plt.title("Confusion Matrix (Darker Blue = More Customers)")
plt.show()

# Get importance scores
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for better plotting
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title("Top 10 Drivers of Churn/Fraud")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.show()