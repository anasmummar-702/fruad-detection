import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. LOAD THE DATA
df = pd.read_csv('BankChurners.csv')

# 2. DATA CLEANING (Drop the noise columns)
cols_to_drop = [
    'CLIENTNUM', 
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
]
df = df.drop(columns=cols_to_drop, errors='ignore')

# --- THE FIX STARTS HERE ---

# 3. INSPECT THE RAW DATA (See what is actually in the column)
print("--- Raw Values in Excel ---")
print(df['Attrition_Flag'].unique())

# 4. CLEAN AND MAP THE TARGET
# .str.strip() removes hidden spaces from the start and end
df['Attrition_Flag'] = df['Attrition_Flag'].astype(str).str.strip()

# Create the map explicitly
target_map = {'Existing Customer': 0, 'Attrited Customer': 1}

# Map the values
df['Attrition_Flag'] = df['Attrition_Flag'].map(target_map)

# 5. VERIFY THE MAPPING
print("\n--- Values After Mapping (Should be 0 and 1) ---")
print(df['Attrition_Flag'].value_counts())

# SAFETY CHECK: If we still have NaNs (values that didn't match), drop them
df = df.dropna(subset=['Attrition_Flag'])

# Ensure it is an integer
df['Attrition_Flag'] = df['Attrition_Flag'].astype(int)

# --- RESUME NORMAL PROCESSING ---

# 6. ENCODE OTHER CATEGORICAL FEATURES
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 7. SPLIT AND APPLY SMOTE
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']

# Check one last time before SMOTE
if len(y.unique()) < 2:
    print("\nCRITICAL ERROR: We still only have 1 class. Check the print outputs above!")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nSUCCESS! SMOTE worked.")
    print(f"Original Training Size: {X_train.shape}")
    print(f"New Balanced Training Size: {X_train_resampled.shape}")