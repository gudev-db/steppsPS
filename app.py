import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import tempfile
import os

# Carregar o modelo treinado
model = YOLO("classW.pt")

# Função para processar o vídeo e classificar cada frame
def process_video(video_file):
    # Salvar o vídeo temporariamente
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    # Abrir o vídeo
    cap = cv2.VideoCapture(temp_file_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Pega a taxa de quadros do vídeo
    frame_count = 0
    class_durations = {}

    # Iniciar a exibição do vídeo no Streamlit
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Realizar a classificação no frame
        results = model(frame)  # Classificação do frame

        # Verificar se results contém os dados de probabilidade
        if hasattr(results, 'probs') and results.probs is not None:
            for class_id, prob in enumerate(results.probs[0]):  # results.probs[0] contém as probabilidades para o primeiro frame
                class_name = results.names[class_id]
                confidence = prob.item()  # Probabilidade da classe

                # Definir um limiar de confiança para considerar a classe como detectada
                if confidence > 0.1:  # Limite de 10%, você pode ajustar conforme necessário
                    timestamp = frame_count / frame_rate  # Calcular o timestamp baseado no número de quadros
                    if class_name not in class_durations:
                        class_durations[class_name] = {'start': timestamp, 'duration': 0}
                    class_durations[class_name]['duration'] += 1 / frame_rate  # Incrementar a duração da classe

                    # Adicionar o nome da classe no frame
                    cv2.putText(frame, f"{class_name}: {confidence*100:.2f}%", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mostrar o frame no Streamlit
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()

    # Remover o arquivo temporário
    os.remove(temp_file_path)

    return class_durations

# Interface do Streamlit
st.title("Classificador de Vídeo com YOLO")
uploaded_file = st.file_uploader("Escolha um arquivo de vídeo", type=["mp4", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Processando vídeo...")
    start_time = time.time()
    durations = process_video(uploaded_file)
    elapsed_time = time.time() - start_time
    st.write(f"Processamento concluído em {elapsed_time:.2f} segundos.")
    st.write("Duração de cada classe detectada (em segundos):")
    
    # Exibir as durações das classes detectadas
    for class_name, times in durations.items():
        st.write(f"{class_name}: {times['duration']:.2f} segundos")
