import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --- SESSION STATE ---
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'selected_id' not in st.session_state:
    st.session_state['selected_id'] = None

# --- 1. CACHED FUNCTION (Model Training & Stats) ---
@st.cache_resource
def load_train_and_scan():
    # 1. Load Data
    try:
        df = pd.read_csv('BankChurners.csv')
    except FileNotFoundError:
        st.error("Error: BankChurners.csv not found.")
        return None, None, None, None, None, None, None, None

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
    
    # Calculate Averages for Safe Customers (For Logic Explanation)
    safe_customers = df[df['Attrition_Flag'] == 0]
    avg_stats = {
        'Trans_Ct': safe_customers['Total_Trans_Ct'].mean(),
        'Rev_Bal': safe_customers['Total_Revolving_Bal'].mean(),
        'Inactive': safe_customers['Months_Inactive_12_mon'].mean(),
        'Util_Ratio': safe_customers['Avg_Utilization_Ratio'].mean()
    }
    
    return rf_model, df_display, X_raw, high_risk_list, X_test, y_test, X_for_train.columns, avg_stats

# --- HELPER: GENERATE TEXT EXPLANATION ---
def generate_explanation(row_data, avg_stats):
    reasons = []
    if row_data['Total_Trans_Ct'] < (avg_stats['Trans_Ct'] * 0.6):
        reasons.append(f"📉 **Drop in Activity:** Transaction Count ({int(row_data['Total_Trans_Ct'])}) is significantly lower than the safe average ({int(avg_stats['Trans_Ct'])}).")
    if row_data['Total_Revolving_Bal'] < 500:
        reasons.append(f"💳 **Zero/Low Balance:** Revolving Balance (${int(row_data['Total_Revolving_Bal'])}) indicates they stopped using the credit line.")
    if row_data['Months_Inactive_12_mon'] >= 3:
        reasons.append(f"💤 **High Inactivity:** Customer has been inactive for {int(row_data['Months_Inactive_12_mon'])} months.")
    if row_data['Avg_Utilization_Ratio'] < 0.1:
        reasons.append(f"📊 **Low Utilization:** Utilization Ratio ({row_data['Avg_Utilization_Ratio']:.2f}) is extremely low compared to active users.")
    if not reasons:
        reasons.append("⚠️ Complex Pattern: The combination of Age, Income, and Spending creates a high-risk profile based on historical data.")
    return reasons

# --- HELPER: PROCESS NEW INPUT ---
def process_new_input(input_dict, original_df, feature_columns):
    input_df = pd.DataFrame([input_dict])
    clean_original = original_df.drop(columns=['Attrition_Flag', 'CLIENTNUM'] + [c for c in original_df.columns if 'Naive' in c], errors='ignore')
    input_df = input_df[clean_original.columns]
    combined_df = pd.concat([clean_original, input_df], axis=0)
    cat_cols = combined_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))
    processed_input = combined_df.iloc[[-1]]
    return processed_input

# --- HELPER: DETERMINE RISK LABEL ---
def get_risk_label(probability):
    if probability >= 0.80: return "Extreme Risk"
    elif probability >= 0.60: return "High Risk"
    elif probability >= 0.50: return "Intermediate Risk"
    else: return "-Safe"

# --- HELPER: INTERACTIVE IMPORTANCE CHART ---
def create_interactive_importance_chart(model, feature_names, client_data_row):
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # Add actual client values for hover
    client_values = []
    for f in feature_names:
        if f in client_data_row:
            client_values.append(client_data_row[f])
        else:
            client_values.append("N/A")
    feature_df['Client Value'] = client_values
    
    feature_df = feature_df.sort_values(by='Importance', ascending=True).tail(10)
    
    fig = px.bar(
        feature_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Reds',
        text='Client Value', hover_data={'Importance': ':.3f', 'Client Value': True, 'Feature': False}
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Importance Impact", yaxis_title=None, showlegend=False, height=400)
    fig.update_traces(textposition='outside')
    return fig

# --- HELPER: INTERACTIVE CONFUSION MATRIX (NEW) ---
def create_interactive_confusion_matrix(model, X_test, y_test):
    # 1. Predict
    y_pred = model.predict(X_test)
    
    # 2. Calculate Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Create Labels
    labels = ['Existing', 'Attrited']
    
    # 4. Plotly Heatmap
    # We use px.imshow which is perfect for matrices
    fig = px.imshow(
        cm,
        text_auto=True, # This automatically writes the numbers (1651, 48, etc)
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues' # Matches your previous style
    )
    
    fig.update_layout(
        title="",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=400
    )
    return fig

# --- MAIN APP ---
rf_model, df_display, X_encoded, high_risk_df, X_test, y_test, train_cols, avg_stats = load_train_and_scan()

if rf_model is not None:
    
    st.title("💳 Fraud & Attrition Detection System")

    # --- MODE SELECTION ---
    st.sidebar.header("🛠️ Dashboard Mode")
    app_mode = st.sidebar.radio("Choose Operation:", ["🔍 Search Existing Database", "📝 Simulate New Customer"])

    # =========================================================
    # MODE 1: SEARCH EXISTING DATABASE
    # =========================================================
    if app_mode == "🔍 Search Existing Database":
        
        st.markdown("### 🚨 High Risk Alert List")
        st.dataframe(
            high_risk_df.style.format({'Risk Score': '{:.2%}'}).background_gradient(subset=['Risk Score'], cmap='Reds'),
            use_container_width=True,
            height=200
        )
        st.info("💡 Copy a Client ID from above and search on the left.")
        st.markdown("---")

        st.sidebar.markdown("---")
        st.sidebar.header("Search Parameters")
        all_ids = df_display['CLIENTNUM'].unique()
        sorted_ids = list(high_risk_df['Client ID']) + list(set(all_ids) - set(high_risk_df['Client ID']))
        
        user_selected_id = st.sidebar.selectbox("Select Client ID:", sorted_ids)

        if st.sidebar.button("Analyze This Case 🚀"):
            st.session_state['analyzed'] = True
            st.session_state['selected_id'] = user_selected_id

        if st.session_state['analyzed'] and st.session_state['selected_id'] is not None:
            selected_id = st.session_state['selected_id']
            
            customer_row = df_display[df_display['CLIENTNUM'] == selected_id].iloc[0]
            encoded_row = X_encoded[X_encoded['CLIENTNUM'] == selected_id].drop('CLIENTNUM', axis=1)
            
            prediction = rf_model.predict(encoded_row)[0]
            probability = rf_model.predict_proba(encoded_row)[0][1]

            st.subheader(f"Case File: Client #{selected_id}")
            col1, col2 = st.columns([1, 2])
            with col1:
                risk_label = get_risk_label(probability)
                if prediction == 1:
                    st.error("⚠️ PREDICTION: ATTRITED / FRAUD")
                    st.metric("Risk Score", f"{probability:.1%}", delta=risk_label)
                    st.markdown("#### 🧠 AI Explanation Pattern")
                    reasons = generate_explanation(customer_row, avg_stats)
                    for r in reasons: st.write(r)
                else:
                    st.success("✅ PREDICTION: SAFE / EXISTING")
                    st.metric("Risk Score", f"{probability:.1%}", delta=risk_label)
                    st.write("Customer behavior aligns with active/loyal profiles.")

            with col2:
                st.write("#### Customer Profile")
                c1, c2, c3 = st.columns(3)
                c1.info(f"**Age:** {customer_row['Customer_Age']}")
                c1.info(f"**Tx Count:** {customer_row['Total_Trans_Ct']}")
                c2.info(f"**Education:** {customer_row['Education_Level']}")
                c2.info(f"**Rev Bal:** ${customer_row['Total_Revolving_Bal']}")
                c3.info(f"**Income:** {customer_row['Income_Category']}")
                c3.info(f"**Limit:** ${customer_row['Credit_Limit']}")

            st.markdown("---")
            col_features, col_matrix = st.columns(2)
            with col_features:
                st.markdown("**🔍 Top Risk Factors (Specific to Client)**")
                st.caption("Hover over bars to see Client's actual data")
                fig = create_interactive_importance_chart(rf_model, encoded_row.columns, customer_row)
                st.plotly_chart(fig, use_container_width=True)
                
            with col_matrix:
                st.markdown("**📈 Global Model Accuracy**")
                # --- NEW INTERACTIVE MATRIX CALL ---
                fig2 = create_interactive_confusion_matrix(rf_model, X_test, y_test)
                st.plotly_chart(fig2, use_container_width=True)

    # =========================================================
    # MODE 2: SIMULATE NEW CUSTOMER
    # =========================================================
    elif app_mode == "📝 Simulate New Customer":
        st.subheader("📝 Simulate Unseen Data")
        st.write("Enter details for a hypothetic customer to test the model.")
        
        with st.form("new_customer_form"):
            colA, colB, colC = st.columns(3)
            with colA:
                age = st.number_input("Customer Age", 20, 80, 45)
                gender = st.selectbox("Gender", df_display['Gender'].unique())
                dependent_count = st.number_input("Dependent Count", 0, 10, 2)
                edu_level = st.selectbox("Education Level", df_display['Education_Level'].unique())
                marital = st.selectbox("Marital Status", df_display['Marital_Status'].unique())
                income = st.selectbox("Income Category", df_display['Income_Category'].unique())
            with colB:
                card_cat = st.selectbox("Card Category", df_display['Card_Category'].unique())
                months_book = st.number_input("Months on Book", 10, 60, 36)
                total_rel_count = st.number_input("Total Relationship Count", 1, 6, 4)
                months_inactive = st.number_input("Months Inactive (12 mon)", 0, 12, 2)
                contacts_count = st.number_input("Contacts Count (12 mon)", 0, 12, 3)
                credit_limit = st.number_input("Credit Limit", 1000.0, 40000.0, 5000.0)
            with colC:
                revolving_bal = st.number_input("Total Revolving Bal", 0, 3000, 1000)
                avg_open_buy = st.number_input("Avg Open To Buy", 0.0, 40000.0, 1000.0)
                amt_chng_q4_q1 = st.number_input("Total Amt Chng Q4/Q1", 0.0, 4.0, 0.7)
                total_trans_amt = st.number_input("Total Trans Amt", 0, 20000, 4000)
                total_trans_ct = st.number_input("Total Trans Ct", 0, 150, 60)
                ct_chng_q4_q1 = st.number_input("Total Ct Chng Q4/Q1", 0.0, 4.0, 0.7)
                avg_util_ratio = st.number_input("Avg Utilization Ratio", 0.0, 1.0, 0.3)
            
            submit_val = st.form_submit_button("Predict Status 🎲")
            
        if submit_val:
            input_data = {
                'Customer_Age': age, 'Gender': gender, 'Dependent_count': dependent_count,
                'Education_Level': edu_level, 'Marital_Status': marital, 'Income_Category': income,
                'Card_Category': card_cat, 'Months_on_book': months_book, 
                'Total_Relationship_Count': total_rel_count, 'Months_Inactive_12_mon': months_inactive,
                'Contacts_Count_12_mon': contacts_count, 'Credit_Limit': credit_limit,
                'Total_Revolving_Bal': revolving_bal, 'Avg_Open_To_Buy': avg_open_buy,
                'Total_Amt_Chng_Q4_Q1': amt_chng_q4_q1, 'Total_Trans_Amt': total_trans_amt,
                'Total_Trans_Ct': total_trans_ct, 'Total_Ct_Chng_Q4_Q1': ct_chng_q4_q1,
                'Avg_Utilization_Ratio': avg_util_ratio
            }
            
            processed_row = process_new_input(input_data, df_display, train_cols)
            prediction = rf_model.predict(processed_row)[0]
            probability = rf_model.predict_proba(processed_row)[0][1]
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                risk_label = get_risk_label(probability)
                st.write("### 🤖 Model Prediction")
                if prediction == 1:
                    st.error("⚠️ STATUS: ATTRITED / FRAUD")
                    st.metric("Confidence", f"{probability:.1%}", delta=risk_label)
                    st.markdown("#### 🧠 Why did the AI say this?")
                    reasons = generate_explanation(input_data, avg_stats)
                    for r in reasons: st.write(r)
                else:
                    st.success("✅ STATUS: SAFE / EXISTING")
                    st.metric("Confidence", f"{1-probability:.1%}", delta=risk_label)
                    st.write("This profile looks like a normal, active customer.")
            with col2:
                st.write("### 📊 Key Indicator Check")
                fig = create_interactive_importance_chart(rf_model, train_cols, input_data)
                st.plotly_chart(fig, use_container_width=True)

    # --- PIE CHART (Common) ---
    st.markdown("---")
    st.subheader("📉 Global Dataset Overview")
    counts = df_display['Attrition_Flag'].value_counts()
    labels = counts.index
    values = counts.values
    colors = ['#1b5e20' if "Existing" in label else '#b71c1c' for label in labels]
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, values=values, pull=[0, 0.1], hole=0.3,
        marker=dict(colors=colors, line=dict(color='#000000', width=2))
    )])
    fig_pie.update_layout(title_text="Total Distribution: Safe vs Attrited", annotations=[dict(text='Total', x=0.5, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig_pie, use_container_width=True)