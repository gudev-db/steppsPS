import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Configuração mínima da página
st.set_page_config(page_title="Detector de Anel de Tensão", layout="centered")
st.title("👁️ Detecção de Anel de Tensão")

# Carregar modelo (cache para melhor performance)
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")  # Substitua pelo seu modelo
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None

model = load_model()

# Upload da imagem
uploaded_file = st.file_uploader("Carregue imagem da íris:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Converter para array numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Fazer predição
    results = model.predict(img, conf=0.5)
    
    # Processar resultados
    if len(results[0].boxes) > 0:
        class_id = int(results[0].boxes[0].cls)
        conf = float(results[0].boxes[0].conf)
        
        # Mostrar resultado
        if class_id == 0:  # True = Anel presente
            st.error(f"🚨 Anel de Tensão DETECTADO (confiança: {conf:.2f})")
            annotated_img = results[0].plot()  # Imagem com bbox
            st.image(annotated_img, caption="Íris com anel de tensão", use_column_width=True)
        else:
            st.success(f"✅ Anel de Tensão NÃO DETECTADO (confiança: {conf:.2f})")
            st.image(img, caption="Íris sem anel de tensão", use_column_width=True)
    else:
        st.warning("Nenhum anel de tensão detectado")
        st.image(img, caption="Íris analisada", use_column_width=True)
