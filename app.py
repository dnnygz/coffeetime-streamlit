from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CoffeeTime Café", page_icon="☕")
st.title("☕ CoffeeTime Café")
st.subheader("Predicción de retorno de clientes")
st.write("Estima si un cliente regresará durante los próximos 30 días.")

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).resolve().parent / "model.joblib")

model = load_model()

with st.form("customer_form"):
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Edad", 18, 100, 30)
        visitas = st.number_input("Visitas últimos 90 días", min_value=0, value=5)
        gasto = st.number_input("Gasto promedio", min_value=0.0, value=25.0)
        dias = st.number_input("Días desde última visita", min_value=0, value=10)
        descuentos = st.number_input("Descuentos utilizados", min_value=0, value=1)
    with c2:
        calificacion = st.number_input("Calificación promedio", 1.0, 5.0, 4.0)
        frecuencia = st.number_input("Frecuencia de compras", min_value=0.0, value=2.0)
        canal = st.selectbox("Canal de compra", ["Presencial", "Aplicación", "Delivery"])
        producto = st.selectbox("Producto favorito", ["Café", "Postres", "Bebidas frías", "Otros"])
        fidelidad = st.selectbox("¿Usa programa de fidelidad?", ["Sí", "No"])
    submitted = st.form_submit_button("Predecir retorno")

if submitted:
    cliente = pd.DataFrame([{
        "edad": edad,
        "visitas_ultimos_90_dias": visitas,
        "gasto_promedio": gasto,
        "dias_desde_ultima_visita": dias,
        "canal_compra": canal,
        "producto_favorito": producto,
        "usa_programa_fidelidad": int(fidelidad == "Sí"),
        "descuentos_utilizados": descuentos,
        "calificacion_promedio": calificacion,
        "frecuencia_compras": frecuencia
    }])
    pred = int(model.predict(cliente)[0])
    prob = float(model.predict_proba(cliente)[0][1])
    st.divider()
    st.metric("Probabilidad estimada de retorno", f"{prob:.1%}")
    if pred == 1:
        st.success("El modelo estima que el cliente REGRESARÁ.")
    else:
        st.warning("El modelo estima que el cliente NO REGRESARÁ.")
    st.caption("Es una estimación del modelo, no una garantía.")

st.caption("Actividad 2 · Gomez Rodriguez Dayana Kety")
