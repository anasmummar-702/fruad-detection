import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Fraud/Churn Detector", layout="wide")

# --- 1. CACHED FUNCTION (Runs once to train model) ---
@st.cache_resource
def load_and_train_model():
    print("Training Model... Please wait.")
    
    # Load Data
    try:
        df = pd.read_csv('BankChurners.csv')
    except FileNotFoundError:
        st.error("Error: BankChurners.csv not found. Please put the file in the same folder.")
        return None, None, None, None, None

    # Keep a copy of original data for display
    df_display = df.copy()

    # Cleaning
    cols_to_drop = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Fix Target
    df['Attrition_Flag'] = df['Attrition_Flag'].astype(str).str.strip()
    target_map = {'Existing Customer': 0, 'Attrited Customer': 1}
    df['Attrition_Flag'] = df['Attrition_Flag'].map(target_map)
    df = df.dropna(subset=['Attrition_Flag'])
    df['Attrition_Flag'] = df['Attrition_Flag'].astype(int)

    # Encode Features (Save Encoders to decode later if needed)
    le_dict = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Split & Smote
    X = df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1) # Drop ID for training
    y = df['Attrition_Flag']
    
    # Store ID mapping to find rows later
    id_map = df['CLIENTNUM'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    return rf_model, X_test, y_test, df_display, X # Return X to get column names

# Load the model (This happens only once)
rf_model, X_test, y_test, df_display, X_features = load_and_train_model()

if rf_model is not None:
    
    # --- SIDEBAR (THE SEARCH BAR) ---
    st.sidebar.header("🔍 Search Customer")
    st.sidebar.write("Simulate dragging/searching for a client:")
    
    # Search Box logic
    # We take a sample of IDs from the file to let you "Search"
    all_client_ids = df_display['CLIENTNUM'].unique()
    selected_id = st.sidebar.selectbox("Select/Type Client ID:", all_client_ids)
    
    run_analysis = st.sidebar.button("Analyze This Customer 🚀")

    # --- MAIN PAGE (THE DETAIL PAGE) ---
    st.title("💳 Credit Card Fraud/Churn Detection System")
    st.markdown("---")

    if run_analysis:
        # 1. Get Customer Data
        customer_row = df_display[df_display['CLIENTNUM'] == selected_id].iloc[0]
        
        # Display Customer Details
        st.subheader(f"📄 Detail Page for Client #{selected_id}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Age:** {customer_row['Customer_Age']}")
            st.info(f"**Gender:** {customer_row['Gender']}")
        with col2:
            st.info(f"**Card Type:** {customer_row['Card_Category']}")
            st.info(f"**Dependent Count:** {customer_row['Dependent_count']}")
        with col3:
            st.info(f"**Total Trans Ct:** {customer_row['Total_Trans_Ct']}")
            st.info(f"**Revolving Bal:** ${customer_row['Total_Revolving_Bal']}")

        # 2. Prepare Data for Prediction (Manual Encoding matching training)
        # We need to recreate the row exactly as the model expects it (Numeric)
        # Note: For a production app, we would use the saved encoders. 
        # Here we assume the model is loaded and we grab the processed X_test row corresponding to this ID if available,
        # or we re-encode. To keep it simple for this demo, we will find the customer in the encoded dataset.
        
        # Find the index of this customer in the dataframe
        idx = df_display.index[df_display['CLIENTNUM'] == selected_id][0]
        
        # Since we processed the whole dataframe in the function, we need to be careful.
        # For this demo, let's predict on the processed features matching that index.
        # We need to re-encode this specific row to match the model inputs.
        
        # (Quick hack for demo: We rely on the fact that we can't easily map back to X_test 
        # because of the split, so we will use the logic that we can re-process this single row).
        
        # Let's use a Dummy Prediction for the UI flow demonstration 
        # (Real implementation requires saving the LabelEncoders globally)
        
        # GET PREDICTION
        # We simply find the row in the X_features dataframe we returned earlier
        input_data = X_features.loc[[idx]]
        prediction = rf_model.predict(input_data)[0]
        probability = rf_model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("🤖 AI Analysis Result")

        # 3. Logic for Display
        if prediction == 1:
            st.error("⚠️ ALERT: High Risk of Attrition / Fraud")
            st.metric(label="Risk Probability", value=f"{probability * 100:.2f}%", delta="-High Risk")
        else:
            st.success("✅ STATUS: Safe / Loyal Customer")
            st.metric(label="Safety Probability", value=f"{(1-probability) * 100:.2f}%", delta="Safe")

        # 4. Explainability Graphs
        st.markdown("---")
        st.subheader("📊 Why did the AI make this decision?")
        
        tab1, tab2 = st.tabs(["Top Factors", "Model Performance"])
        
        with tab1:
            # Feature Importance Plot
            importances = rf_model.feature_importances_
            feature_names = X_features.columns
            feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_df = feature_df.sort_values(by='Importance', ascending=False).head(8)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis', ax=ax)
            st.pyplot(fig)
            st.caption("These are the behaviors that most strongly influence the decision.")

        with tab2:
            # Confusion Matrix Plot
            fig2, ax2 = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', ax=ax2, display_labels=["Existing", "Attrited"])
            st.pyplot(fig2)

    else:
        st.info("👈 Please select a Customer ID from the Sidebar and click 'Analyze' to see the Detail Page.")
        
        # Show a preview of data on the landing page
        st.subheader("Dataset Preview")
        st.dataframe(df_display.head())