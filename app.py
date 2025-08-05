import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Iridologia - Detecção de Anel de Tensão",
    page_icon="👁️",
    layout="wide"
)

# Barra lateral
with st.sidebar:
    st.title("Configurações")
    confidence = st.slider("Limite de Confiança", 0.0, 1.0, 0.5, 0.01)
    model_path = st.text_input("Caminho do Modelo", "best.pt")
    st.markdown("---")
    st.markdown("### Como Usar")
    st.info("1. Faça upload de uma imagem da íris\n2. O sistema detectará automaticamente\n3. Veja o resultado e diagnóstico")
    st.markdown("---")
    st.markdown("### Sobre o Anel de Tensão")
    st.warning("""
    O anel de tensão (também chamado de anel neurovascular) indica:
    - **Presente**: Possível estresse crônico ou tensão acumulada
    - **Ausente**: Níveis normais de tensão""")

# Carregar modelo
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

model = load_model(model_path)

# Conteúdo principal
st.title("👁️ Análise Iridológica - Detecção de Anel de Tensão")
st.markdown("""
Este sistema utiliza inteligência artificial para identificar a presença do anel de tensão em imagens da íris,
um importante marcador na análise iridológica.
""")

# Upload de imagem
uploaded_file = st.file_uploader(
    "Carregue uma foto da íris:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Processar imagem
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Exibir original
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Íris Original")
        st.image(image, caption="Sua imagem", use_column_width=True)
    
    # Salvar temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        
        if model is not None:
            with st.spinner("Analisando íris..."):
                try:
                    # Predição
                    results = model.predict(
                        source=temp_file.name,
                        conf=confidence,
                        save=False
                    )
                    
                    # Processar resultados
                    with col2:
                        st.subheader("Resultado da Análise")
                        
                        for result in results:
                            # Desenhar bounding box
                            annotated_img = img_array.copy()
                            if result.boxes:
                                for box in result.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Mostrar imagem com anotação
                            st.image(
                                annotated_img if result.boxes else img_array,
                                caption="Íris Analisada",
                                use_column_width=True
                            )
                            
                            # Diagnóstico
                            if len(result.boxes) > 0:
                                class_id = int(result.boxes[0].cls)
                                conf = float(result.boxes[0].conf)
                                
                                if class_id == 0:  # True = Anel presente
                                    st.error(f"🚨 **Resultado**: Anel de Tensão Detectado (confiança: {conf:.2f})")
                                    st.warning("""
                                    **Interpretação Iridológica:**
                                    - Possível estresse crônico
                                    - Tensão no sistema nervoso
                                    - Acúmulo de toxinas""")
                                else:  # False = Anel ausente
                                    st.success(f"✅ **Resultado**: Sem Anel de Tensão Detectado (confiança: {conf:.2f})")
                                    st.info("""
                                    **Interpretação Iridológica:**
                                    - Níveis normais de tensão
                                    - Sistema neurovegetativo equilibrado""")
                            else:
                                st.warning("Nenhum anel de tensão detectado na imagem")
                    
                    # Limpar arquivo temporário
                    os.unlink(temp_file.name)
                    
                except Exception as e:
                    st.error(f"Erro na análise: {e}")
else:
    st.info("Por favor, carregue uma imagem da íris para análise")

