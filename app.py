import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --- SESSION STATE SETUP (The Fix for the "Reset" issue) ---
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'selected_id' not in st.session_state:
    st.session_state['selected_id'] = None

# --- 1. CACHED FUNCTION (Model Training) ---
@st.cache_resource
def load_train_and_scan():
    # 1. Load Data
    try:
        df = pd.read_csv('BankChurners.csv')
    except FileNotFoundError:
        st.error("Error: BankChurners.csv not found.")
        return None, None, None, None, None, None

    df_display = df.copy()

    # 2. Cleaning
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

    # 3. Encode
    X_raw = df.drop('Attrition_Flag', axis=1)
    categorical_cols = X_raw.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col])

    # 4. Train
    X_for_train = X_raw.drop('CLIENTNUM', axis=1)
    y = df['Attrition_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X_for_train, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # 5. Scan for Fraud
    all_probs = rf_model.predict_proba(X_for_train)[:, 1]
    
    risk_df = pd.DataFrame({
        'Client ID': df_display['CLIENTNUM'],
        'Risk Score': all_probs
    })
    
    high_risk_list = risk_df[risk_df['Risk Score'] > 0.5].sort_values(by='Risk Score', ascending=False)
    
    return rf_model, df_display, X_raw, high_risk_list, X_test, y_test

# --- MAIN APP ---
rf_model, df_display, X_encoded, high_risk_df, X_test, y_test = load_train_and_scan()

if rf_model is not None:
    
    st.title("💳 Fraud & Attrition Detection System")
    
    # --- ALERT SECTION ---
    st.markdown("### 🚨 High Risk Alert List")
    st.dataframe(
        high_risk_df.style.format({'Risk Score': '{:.2%}'}).background_gradient(subset=['Risk Score'], cmap='Reds'),
        use_container_width=True,
        height=200
    )
    st.info("💡 Copy a Client ID from above and search on the left.")
    st.markdown("---")

    # --- SIDEBAR ---
    st.sidebar.header("🔍 Investigate Customer")
    all_ids = df_display['CLIENTNUM'].unique()
    sorted_ids = list(high_risk_df['Client ID']) + list(set(all_ids) - set(high_risk_df['Client ID']))
    
    # Select box
    user_selected_id = st.sidebar.selectbox("Select Client ID:", sorted_ids)

    # Button
    if st.sidebar.button("Analyze This Case 🚀"):
        st.session_state['analyzed'] = True
        st.session_state['selected_id'] = user_selected_id

    # --- MAIN ANALYSIS AREA ---
    # We check session_state to ensure the page persists even if you interact with charts
    if st.session_state['analyzed']:
        
        selected_id = st.session_state['selected_id']
        
        # --- DETAIL PAGE ---
        customer_row = df_display[df_display['CLIENTNUM'] == selected_id].iloc[0]
        encoded_row = X_encoded[X_encoded['CLIENTNUM'] == selected_id].drop('CLIENTNUM', axis=1)
        
        prediction = rf_model.predict(encoded_row)[0]
        probability = rf_model.predict_proba(encoded_row)[0][1]

        # Result Display
        st.subheader(f"Case File: Client #{selected_id}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("⚠️ PREDICTION: FRAUD / ATTRITED")
                st.metric("Risk Score", f"{probability:.1%}", delta="High Risk")
            else:
                st.success("✅ PREDICTION: SAFE / EXISTING")
                st.metric("Risk Score", f"{probability:.1%}", delta="-Safe")
                
        with col2:
            st.write("#### Customer Profile")
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Age:** {customer_row['Customer_Age']}")
            c1.info(f"**Gender:** {customer_row['Gender']}")
            c2.info(f"**Education:** {customer_row['Education_Level']}")
            c2.info(f"**Income:** {customer_row['Income_Category']}")
            c3.info(f"**Tx Count:** {customer_row['Total_Trans_Ct']}")
            c3.info(f"**Revolving Bal:** ${customer_row['Total_Revolving_Bal']}")

        # --- DUAL CHART SECTION (Side by Side) ---
        st.markdown("---")
        st.subheader("📊 AI Analysis & Reasoning")
        
        # Create 2 Columns for the charts
        col_features, col_matrix = st.columns(2)
        
        # LEFT COLUMN: AI REASONING (Feature Importance)
        with col_features:
            st.markdown("**🔍 Top Risk Factors (Specific to Client)**")
            st.caption("Behaviors that flagged this specific customer.")
            
            importances = rf_model.feature_importances_
            feature_names = encoded_row.columns
            feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_df = feature_df.sort_values(by='Importance', ascending=False).head(8)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Importance', y='Feature', data=feature_df, palette='Reds_r', ax=ax)
            st.pyplot(fig)

        # RIGHT COLUMN: MODEL ACCURACY (Confusion Matrix)
        with col_matrix:
            st.markdown("**📈 Global Model Accuracy**")
            st.caption("How accurately the model detects fraud in testing.")
            
            # This generates the image you provided
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', ax=ax2, display_labels=["Existing", "Attrited"])
            plt.title("Confusion Matrix")
            st.pyplot(fig2)

    # --- PIE CHART (Global Overview at Bottom) ---
    st.markdown("---")
    st.subheader("📉 Global Dataset Overview")
    
    counts = df_display['Attrition_Flag'].value_counts()
    labels = counts.index
    values = counts.values
    
    # 3D Pie Chart Colors (Dark Green / Dark Red)
    colors = []
    for label in labels:
        if "Existing" in label:
            colors.append('#1b5e20') # Dark Green
        else:
            colors.append('#b71c1c') # Dark Red

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        pull=[0, 0.1], 
        hole=0.3,
        marker=dict(colors=colors, line=dict(color='#000000', width=2))
    )])
    
    fig_pie.update_layout(
        title_text="Total Distribution: Safe vs Attrited",
        annotations=[dict(text='Total', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)