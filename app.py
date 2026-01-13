import streamlit as st
import pandas as pd
import joblib

# =============================
# Load model & data
# =============================
model = joblib.load('model.pkl')
df = pd.read_csv('data_biostat_fix.csv')

# =============================
# Title
# =============================
st.title("Dashboard Prediksi Penyakit Jantung")
st.write("Metode Regresi Logistik (Data Biostatistik)")

# =============================
# Sidebar Input
# =============================
st.sidebar.header("Input Data Pasien")

age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (0=Perempuan, 1=Laki-laki)", [0, 1])
restingbp = st.sidebar.number_input("RestingBP", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fastingbs = st.sidebar.selectbox("FastingBS", [0, 1])
maxhr = st.sidebar.number_input("MaxHR", 60, 220, 150)
exerciseangina = st.sidebar.selectbox("ExerciseAngina", [0, 1])

# =============================
# Prediction
# =============================
input_data = pd.DataFrame([[
    age, sex, restingbp, chol, fastingbs, maxhr, exerciseangina
]], columns=[
    'Age', 'Sex', 'RestingBP', 'Cholesterol',
    'FastingBS', 'MaxHR', 'ExerciseAngina'
])

if st.sidebar.button("Prediksi"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Berisiko Penyakit Jantung\n\nProbabilitas: {probability:.2f}")
    else:
        st.success(f"✅ Tidak Berisiko\n\nProbabilitas: {probability:.2f}")

# =============================
# Visualization
# =============================
st.subheader("Distribusi Data Penyakit Jantung")
st.bar_chart(df['HeartDisease'].value_counts())
