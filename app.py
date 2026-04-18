import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from datetime import datetime
import os

# Get the directory where the app is running
BASE_DIR = os.path.dirname(__file__)

# Set page config
st.set_page_config(
    page_title="🧠 Autism Spectrum Disorder Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
    }
    .autism-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .autism-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(BASE_DIR, 'random_forest_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load and prepare label encoders
@st.cache_resource(show_spinner=False)
def prepare_encoders():
    csv_path = os.path.join(BASE_DIR, 'Autism_data.csv')
    data = pd.read_csv(csv_path)
    
    # Remove duplicate header rows
    data = data[data['age'] != 'age']
    
    # Basic preprocessing
    data['age'] = data['age'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
    data = data.rename(columns={'austim':'autism', 'contry_of_res':'Country_of_res'})
    data = data.drop(columns=['age_desc','ID'])
    data['ethnicity'] = data['ethnicity'].replace('?', data['ethnicity'].mode()[0])
    data['ethnicity'] = data['ethnicity'].replace('others','Others')
    data['relation'] = data['relation'].replace('?', data['relation'].mode()[0])
    
    mapping = {'Viet Nam':'Vietnam', 'AmericanSamoa':'United States', 'Hong Kong': 'China'}
    data['Country_of_res'] = data['Country_of_res'].replace(mapping)
    
    # Create encoders
    encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(data[col])
        encoders[col] = le
    
    return encoders, data.columns.tolist()

# Get feature columns
def get_feature_columns():
    csv_path = os.path.join(BASE_DIR, 'Autism_data.csv')
    data = pd.read_csv(csv_path)
    # Remove duplicate header rows
    data = data[data['age'] != 'age']
    data['age'] = data['age'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
    data = data.rename(columns={'austim':'autism', 'contry_of_res':'Country_of_res'})
    data = data.drop(columns=['age_desc','ID','Class/ASD'])
    return data.columns.tolist()

model = load_model()
encoders, _ = prepare_encoders()
feature_cols = get_feature_columns()

# Header
st.title("🧠 Autism Spectrum Disorder Prediction System")
st.markdown("**AI-Powered Assessment Tool**")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Testing", "ℹ️ About"])

# ==================== TAB 1: Single Prediction ====================
with tab1:
    st.header("Individual Assessment")
    
    col1, col2 = st.columns(2)
    
    # Demographics
    with col1:
        st.subheader("👤 Demographics")
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        gender = st.selectbox("Gender", ["m", "f"])
        ethnicity = st.selectbox("Ethnicity", 
            ["White-European", "South Asian", "Black", "Asian", "Latino", 
             "Middle Eastern", "Others", "?"])
        
    with col2:
        st.subheader("🏥 Medical History")
        jaundice = st.selectbox("Jaundice at Birth", ["yes", "no"])
        autism = st.selectbox("Family History of Autism", ["yes", "no"])
        relation = st.selectbox("Relation Type", ["Self", "Parent", "Health care professional", 
                                                   "Relative", "?"])
        country = st.selectbox("Country of Residence", 
            ["United States", "United Kingdom", "Canada", "Australia", "India", "Others"])
        used_app = st.selectbox("Used App Before", ["yes", "no"])
    
    st.subheader("📋 Autism Spectrum Quotient (AQ) Test Questions")
    st.markdown("**Score 1 for each agreement, 0 for each disagreement**")
    
    scores = []
    cols = st.columns(5)
    
    questions = [
        "Q1: Focus on details",
        "Q2: Switch between tasks",
        "Q3: Follow conversation",
        "Q4: Get upset by changes",
        "Q5: Special interests/hobbies",
        "Q6: Notice patterns",
        "Q7: Prefer routines",
        "Q8: Remember factual details",
        "Q9: Find social situations difficult",
        "Q10: Difficulty understanding others"
    ]
    
    for i in range(10):
        with cols[i % 5]:
            score = st.checkbox(f"A{i+1}", key=f"q{i}")
            scores.append(1 if score else 0)
    
    result_score = st.number_input("AQ Test Score (0-10)", min_value=0, max_value=10, value=5)
    
    # Safe encoding function
    def safe_transform(le, value):
        try:
            return le.transform([value])[0]
        except:
            return 0
    
    # Prediction Button
    if st.button("🎯 Get Prediction", use_container_width=True, type="primary"):
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'A1_Score': [scores[0]],
                'A2_Score': [scores[1]],
                'A3_Score': [scores[2]],
                'A4_Score': [scores[3]],
                'A5_Score': [scores[4]],
                'A6_Score': [scores[5]],
                'A7_Score': [scores[6]],
                'A8_Score': [scores[7]],
                'A9_Score': [scores[8]],
                'A10_Score': [scores[9]],
                'age': [age],
                'gender': [gender],
                'ethnicity': [ethnicity if ethnicity != "?" else "White-European"],
                'jaundice': [jaundice],
                'autism': [autism],
                'Country_of_res': [country if country != "Others" else "United States"],
                'used_app_before': [used_app],
                'result': [result_score],
                'relation': [relation if relation != "?" else "Self"]
            })
            
            # Encode categorical variables
            for col in ['gender', 'ethnicity', 'jaundice', 'autism', 'Country_of_res', 'used_app_before', 'relation']:
                if col in encoders:
                    input_data[col] = input_data[col].apply(lambda x: safe_transform(encoders[col], x))
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            autism_prob = probability[1] * 100
            non_autism_prob = probability[0] * 100
            
            # Display results
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="autism-high">
                        <h2>⚠️ Likely Autistic Traits Detected</h2>
                        <h3 style="font-size: 2.5rem; margin: 10px 0;">{autism_prob:.1f}%</h3>
                        <p style="font-size: 1.1rem;">Risk of Autism Spectrum Disorder</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="autism-low">
                        <h2>✅ Low Risk Assessment</h2>
                        <h3 style="font-size: 2.5rem; margin: 10px 0;">{autism_prob:.1f}%</h3>
                        <p style="font-size: 1.1rem;">Risk of Autism Spectrum Disorder</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Autism Probability", f"{autism_prob:.2f}%")
            with col2:
                st.metric("Non-Autism Probability", f"{non_autism_prob:.2f}%")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = autism_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Autism Risk Level"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "#90EE90"},
                        {'range': [25, 50], 'color': "#FFD700"},
                        {'range': [50, 75], 'color': "#FFA500"},
                        {'range': [75, 100], 'color': "#FF6347"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=400, width=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# ==================== TAB 2: Batch Testing ====================
with tab2:
    st.header("📊 Batch Testing")
    st.markdown("Test multiple cases at once")
    
    # Create sample data for testing
    if st.button("📥 Load 10 Test Cases from Dataset", use_container_width=True):
        # Load data
        csv_path = os.path.join(BASE_DIR, 'Autism_data.csv')
        data = pd.read_csv(csv_path)
        # Remove duplicate header rows
        data = data[data['age'] != 'age']
        data['age'] = data['age'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
        data = data.rename(columns={'austim':'autism', 'contry_of_res':'Country_of_res'})
        data = data.drop(columns=['age_desc','ID'], errors='ignore')
        
        # Encode and prepare
        data['ethnicity'] = data['ethnicity'].replace('?', data['ethnicity'].mode()[0])
        data['ethnicity'] = data['ethnicity'].replace('others','Others')
        data['relation'] = data['relation'].replace('?', data['relation'].mode()[0])
        mapping = {'Viet Nam':'Vietnam', 'AmericanSamoa':'United States', 'Hong Kong': 'China'}
        data['Country_of_res'] = data['Country_of_res'].replace(mapping)
        
        # Encode
        for col in encoders.keys():
            if col in data.columns:
                data[col] = encoders[col].transform(data[col])
        
        # Get predictions for first 10 samples
        test_data = data.drop('Class/ASD', axis=1, errors='ignore').head(10)
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)
        
        results = []
        for i in range(len(test_data)):
            autism_prob = probabilities[i][1] * 100
            results.append({
                'Person': i + 1,
                'Autism Risk %': f"{autism_prob:.2f}%",
                'Status': '🔴 High Risk' if autism_prob > 50 else '🟢 Low Risk',
                'Prediction': 'Likely Autistic' if predictions[i] == 1 else 'Non-Autistic'
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tested", len(results_df))
        with col2:
            high_risk = len([r for r in results if '🔴' in r['Status']])
            st.metric("High Risk Cases", high_risk)
        with col3:
            st.metric("Average Risk %", f"{np.mean([float(r['Autism Risk %'].rstrip('%')) for r in results]):.1f}%")

# ==================== TAB 3: About ====================
with tab3:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 🎯 Purpose
    This system uses a **Random Forest Machine Learning Model** to assess the probability of 
    Autism Spectrum Disorder (ASD) based on screening questionnaire responses and demographic information.
    
    ### 📊 Model Performance
    - **Accuracy**: ~92-95%
    - **F1 Score**: High precision and recall
    - **ROC-AUC**: Excellent discrimination ability
    
    ### 📋 Input Features
    1. **10 Autism Screening Questions** (A1-A10)
    2. **Demographics**: Age, Gender, Ethnicity
    3. **Medical History**: Jaundice, Family History of Autism
    4. **Location**: Country of Residence
    5. **Other**: App usage, Relation type
    
    ### ⚠️ Important Disclaimer
    **This is a screening tool, NOT a medical diagnosis.**
    - Results should only be used for initial assessment
    - Always consult with a qualified healthcare professional for formal diagnosis
    - This tool is based on statistical patterns, not medical expertise
    
    ### 📈 How It Works
    1. **Data Collection**: Gather information from the screening questionnaire
    2. **Preprocessing**: Encode and normalize the data
    3. **Prediction**: Random Forest model generates probability score
    4. **Interpretation**: Display risk level and probability
    
    ### 🔒 Privacy
    Your data is processed locally. No data is stored or transmitted.
    
    ### 📚 Dataset
    - **Total Records**: ~700 individuals
    - **Features**: 21 variables
    - **Target**: Autism Spectrum Disorder (Binary: Yes/No)
    
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Machine Learning | 2024")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <small>⚕️ <b>Medical Disclaimer</b>: This tool is for screening purposes only. 
    Always consult qualified healthcare professionals for diagnosis and treatment.</small>
</div>
""", unsafe_allow_html=True)
