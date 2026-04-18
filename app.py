import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path

# Get the directory where the app is running
BASE_DIR = Path(__file__).resolve().parent

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
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model_path = BASE_DIR / 'random_forest_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# ================= ENCODERS =================
@st.cache_resource
def prepare_encoders():
    csv_path = BASE_DIR / 'Autism_data.csv'
    data = pd.read_csv(csv_path)

    # Remove duplicate header rows
    data = data[data['age'] != 'age']

    # Basic preprocessing
    data['age'] = data['age'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
    data = data.rename(columns={'austim': 'autism', 'contry_of_res': 'Country_of_res'})

    # 🔧 FIX 1: safe drop (avoid crash)
    data = data.drop(columns=['age_desc', 'ID'], errors='ignore')

    data['ethnicity'] = data['ethnicity'].replace('?', data['ethnicity'].mode()[0])
    data['ethnicity'] = data['ethnicity'].replace('others', 'Others')
    data['relation'] = data['relation'].replace('?', data['relation'].mode()[0])

    mapping = {'Viet Nam': 'Vietnam', 'AmericanSamoa': 'United States', 'Hong Kong': 'China'}
    data['Country_of_res'] = data['Country_of_res'].replace(mapping)

    encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(data[col].astype(str))
        encoders[col] = le

    return encoders, data.columns.tolist()

# ================= FEATURES =================
def get_feature_columns():
    csv_path = BASE_DIR / 'Autism_data.csv'
    data = pd.read_csv(csv_path)

    data = data[data['age'] != 'age']
    data['age'] = data['age'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
    data = data.rename(columns={'austim': 'autism', 'contry_of_res': 'Country_of_res'})

    # 🔧 FIX 2: safe drop
    data = data.drop(columns=['age_desc', 'ID', 'Class/ASD'], errors='ignore')

    return data.columns.tolist()

# ================= LOAD =================
model = load_model()
encoders, all_cols = prepare_encoders()
feature_cols = get_feature_columns()

# ================= HEADER =================
st.title("🧠 Autism Spectrum Disorder Prediction System")
st.markdown("**AI-Powered Assessment Tool**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Testing", "ℹ️ About"])

# ================= TAB 1 =================
with tab1:
    st.header("Individual Assessment")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 25)
        gender = st.selectbox("Gender", ["m", "f"])
        ethnicity = st.selectbox("Ethnicity", ["White-European", "South Asian", "Black", "Asian", "Latino", "Middle Eastern", "Others", "?"])

    with col2:
        jaundice = st.selectbox("Jaundice at Birth", ["yes", "no"])
        autism = st.selectbox("Family History", ["yes", "no"])
        relation = st.selectbox("Relation Type", ["Self", "Parent", "Health care professional", "Relative", "?"])
        country = st.selectbox("Country", ["United States", "United Kingdom", "Canada", "Australia", "India", "Others"])
        used_app = st.selectbox("Used App Before", ["yes", "no"])

    scores = []
    cols = st.columns(5)

    for i in range(10):
        with cols[i % 5]:
            score = st.checkbox(f"A{i+1}", key=f"q{i}")
            scores.append(1 if score else 0)

    result_score = st.number_input("AQ Score", 0, 10, 5)

    if st.button("🎯 Predict"):
        try:
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

            # ================= FIX 3: SAFE ENCODING =================
            for col in input_data.columns:
                if col in encoders:
                    try:
                        input_data[col] = encoders[col].transform(input_data[col].astype(str))
                    except:
                        input_data[col] = 0

            # ================= FIX 4: COLUMN ORDER =================
            input_data = input_data.reindex(columns=feature_cols, fill_value=0)

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            autism_prob = probability[1] * 100

            if prediction == 1:
                st.markdown(f"<div class='autism-high'><h2>⚠️ High Risk</h2><h3>{autism_prob:.1f}%</h3></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='autism-low'><h2>✅ Low Risk</h2><h3>{autism_prob:.1f}%</h3></div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(str(e))

# ================= TAB 2 =================
with tab2:
    st.header("Batch Testing")

    if st.button("Run Batch Test"):
        data = pd.read_csv(BASE_DIR / 'Autism_data.csv')

        data = data[data['age'] != 'age']
        data['age'] = data['age'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)

        data = data.rename(columns={'austim': 'autism', 'contry_of_res': 'Country_of_res'})

        data = data.drop(columns=['age_desc', 'ID', 'Class/ASD'], errors='ignore')

        for col in encoders:
            if col in data.columns:
                try:
                    data[col] = encoders[col].transform(data[col].astype(str))
                except:
                    data[col] = 0

        data = data.reindex(columns=feature_cols, fill_value=0).head(10)

        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1] * 100

        df = pd.DataFrame({
            "Prediction": preds,
            "Risk %": probs
        })

        st.dataframe(df)

# ================= TAB 3 =================
with tab3:
    st.info("⚠️ This tool is NOT a medical diagnosis system.")
