import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilita la GPU

import streamlit as st 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Configuración de la página (debe ser lo primero en ejecutarse)
st.set_page_config(page_title="Diagnóstico de COVID-19", layout="wide")

# Verificar si los archivos existen
if not os.path.exists("modelo_covid.h5"):
    st.error("Error: El archivo 'modelo_covid.h5' no se encuentra.")
    model = None
else:
    model = load_model("modelo_covid.h5")

if not os.path.exists("scaler.pkl"):
    st.error("Error: El archivo 'scaler.pkl' no se encuentra.")
    scaler = None
else:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

# Función de predicción
def predict_covid(gender, age_year, fever, cough, runny_nose, pneumonia, diarrhea, lung_infection, travel_history, isolation_treatment):
    try:
        gender = 1 if gender == "Hombre" else 0
        features = np.array([[gender, age_year, fever, cough, runny_nose, pneumonia, diarrhea, lung_infection, travel_history, isolation_treatment]])
        if scaler is not None:
            features = scaler.transform(features)         
        else:
            return "⚠️ Error en el procesamiento: Escalador no disponible ⚠️"
        prediction = model.predict(features)[0][0]
        return "🦠 COVID-19 Positivo 🦠" if prediction > 0.5 else "✅ No tiene COVID-19 ✅"
    except Exception as e:
        return "⚠️ Error en el procesamiento ⚠️"

# Estilos CSS
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        .stButton>button { background-color: #ff4b4b; color: white; font-size: 20px; border-radius: 10px; }
        .stTextInput, .stNumberInput, .stSelectbox, .stCheckbox { font-size: 18px; }
        .result-box { font-size: 22px; font-weight: bold; text-align: center; padding: 10px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
title_html = "<h1 style='text-align: center; color: #333;'>Sistema de Diagnóstico de COVID-19</h1>"
st.markdown(title_html, unsafe_allow_html=True)

st.markdown("---")

# Diseño en dos columnas
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Género", ["Hombre", "Mujer"])
    age_year = st.number_input("Edad", min_value=0, max_value=120, step=1)
    fever = st.checkbox("Fiebre")
    cough = st.checkbox("Tos")
    runny_nose = st.checkbox("Secreción nasal")

with col2:
    pneumonia = st.checkbox("Neumonía")
    diarrhea = st.checkbox("Diarrea")
    lung_infection = st.checkbox("Infección pulmonar")
    travel_history = st.checkbox("Historial de viaje")
    isolation_treatment = st.checkbox("Tratamiento en aislamiento")

st.markdown("---")

# Botón de diagnóstico
if st.button("🔍 Diagnosticar"):
    if model is None or scaler is None:
        st.error("Error: No se pudo cargar el modelo o el escalador correctamente.")
    else:
        result = predict_covid(gender, age_year, fever, cough, runny_nose, pneumonia, diarrhea, lung_infection, travel_history, isolation_treatment)
        result_color = "#ff4b4b" if "Positivo" in result else "#4CAF50"
        st.markdown(f'<div class="result-box" style="background-color: {result_color}; color: white;">{result}</div>', unsafe_allow_html=True)
