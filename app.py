import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from pathlib import Path

# ===================== SETUP =====================
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="🧠 Autism Spectrum Disorder Prediction",
    page_icon="🧠",
    layout="wide"
)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    model_path = BASE_DIR / 'random_forest_model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# ===================== PREPROCESS =====================
def preprocess_data(data):
    data = data[data['age'] != 'age']

    data['age'] = data['age'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)

    data = data.rename(columns={
        'austim': 'autism',
        'contry_of_res': 'Country_of_res'
    })

    data = data.drop(columns=['age_desc', 'ID', 'Class/ASD'], errors='ignore')

    data['ethnicity'] = data['ethnicity'].replace('?', data['ethnicity'].mode()[0])
    data['ethnicity'] = data['ethnicity'].replace('others', 'Others')

    data['relation'] = data['relation'].replace('?', data['relation'].mode()[0])

    mapping = {
        'Viet Nam': 'Vietnam',
        'AmericanSamoa': 'United States',
        'Hong Kong': 'China'
    }

    data['Country_of_res'] = data['Country_of_res'].replace(mapping)

    return data

# ===================== LOAD ENCODERS =====================
@st.cache_resource
def prepare_encoders():
    csv_path = BASE_DIR / 'Autism_data.csv'
    data = pd.read_csv(csv_path)

    data = preprocess_data(data)

    encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(data[col])
        encoders[col] = le

    feature_cols = [col for col in data.columns if col != 'Class/ASD']

    return encoders, feature_cols

# ===================== LOAD =====================
model = load_model()
encoders, feature_cols = prepare_encoders()

# ===================== UI =====================
st.title("🧠 Autism Spectrum Disorder Prediction System")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch", "ℹ️ About"])

# ===================== TAB 1 =====================
with tab1:
    st.header("Individual Prediction")

    age = st.number_input("Age", 0, 100, 25)
    gender = st.selectbox("Gender", ["m", "f"])
    ethnicity = st.selectbox("Ethnicity", ["White-European", "Asian", "Others"])
    jaundice = st.selectbox("Jaundice", ["yes", "no"])
    autism = st.selectbox("Family Autism", ["yes", "no"])
    country = st.selectbox("Country", ["United States", "United Kingdom", "India"])
    used_app = st.selectbox("Used App", ["yes", "no"])
    relation = st.selectbox("Relation", ["Self", "Parent", "Relative"])

    scores = []
    for i in range(10):
        scores.append(st.checkbox(f"A{i+1}", key=i))

    if st.button("Predict"):
        try:
            input_data = pd.DataFrame({
                'A1_Score': [int(scores[0])],
                'A2_Score': [int(scores[1])],
                'A3_Score': [int(scores[2])],
                'A4_Score': [int(scores[3])],
                'A5_Score': [int(scores[4])],
                'A6_Score': [int(scores[5])],
                'A7_Score': [int(scores[6])],
                'A8_Score': [int(scores[7])],
                'A9_Score': [int(scores[8])],
                'A10_Score': [int(scores[9])],
                'age': [age],
                'gender': [gender],
                'ethnicity': [ethnicity],
                'jaundice': [jaundice],
                'autism': [autism],
                'Country_of_res': [country],
                'used_app_before': [used_app],
                'relation': [relation],
                'result': [sum(scores)]
            })

            # Encoding safely
            for col in input_data.columns:
                if col in encoders:
                    try:
                        input_data[col] = encoders[col].transform(input_data[col])
                    except:
                        input_data[col] = 0

            # FIX ORDER 🚨
            input_data = input_data.reindex(columns=feature_cols, fill_value=0)

            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ High Risk: {prob:.2f}%")
            else:
                st.success(f"✅ Low Risk: {prob:.2f}%")

        except Exception as e:
            st.error(str(e))

# ===================== TAB 2 =====================
with tab2:
    st.header("Batch Testing")

    if st.button("Run Batch Test"):
        csv_path = BASE_DIR / 'Autism_data.csv'
        data = pd.read_csv(csv_path)

        data = preprocess_data(data)

        for col in encoders:
            if col in data.columns:
                try:
                    data[col] = encoders[col].transform(data[col])
                except:
                    data[col] = 0

        if 'Class/ASD' in data.columns:
            test_data = data.drop('Class/ASD', axis=1).head(10)
        else:
            test_data = data.head(10)

        test_data = test_data.reindex(columns=feature_cols, fill_value=0)

        preds = model.predict(test_data)
        probs = model.predict_proba(test_data)[:, 1] * 100

        df = pd.DataFrame({
            "Prediction": preds,
            "Risk %": probs
        })

        st.dataframe(df)

# ===================== TAB 3 =====================
with tab3:
    st.info("This is a medical screening tool only — not a diagnosis system.")
