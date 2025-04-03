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

# Cargar modelo y escalador con caché
@st.cache_resource()
def load_resources():
    if not os.path.exists("modelo_covid.h5"):
        st.error("Error: El archivo 'modelo_covid.h5' no se encuentra.")
        return None, None
    model = load_model("modelo_covid.h5")
    
    if not os.path.exists("scaler.pkl"):
        st.error("Error: El archivo 'scaler.pkl' no se encuentra.")
        return model, None
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

model, scaler = load_resources()

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

# Cargar estilos CSS
if os.path.exists("styles.css"):
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Encabezado en HTML
st.markdown("""
    <h1 style='text-align: center; color: #333;'>Sistema de Diagnóstico de COVID-19</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# Diseño en dos columnas
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Género", ["Hombre", "Mujer"])
    age_year = st.text_input("Edad", "")
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

# Validación de entrada
def validate_input():
    if not age_year.isdigit():
        st.error("Por favor, ingrese una edad válida en números.")
        return False
    age = int(age_year)
    if age < 0 or age > 120:
        st.error("Por favor, ingrese una edad entre 0 y 120.")
        return False
    return age

# Botón de diagnóstico
if st.button("🔍 Diagnosticar"):
    age = validate_input()
    if age is not False:
        if model is None or scaler is None:
            st.error("Error: No se pudo cargar el modelo o el escalador correctamente.")
        else:
            result = predict_covid(gender, age, fever, cough, runny_nose, pneumonia, diarrhea, lung_infection, travel_history, isolation_treatment)
            result_color = "#ff4b4b" if "Positivo" in result else "#4CAF50"
            st.markdown(f'<div class="result-box" style="background-color: {result_color}; color: white;">{result}</div>', unsafe_allow_html=True)
