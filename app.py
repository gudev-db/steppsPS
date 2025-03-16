import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Carregar o modelo treinado
model = YOLO('caminho/para/seu_modelo.pt')

# Função para processar o vídeo e classificar cada frame
def process_video(video_file):
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    class_durations = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Realizar a classificação no frame
        results = model(frame)

        # Obter a classe com maior confiança
        if results.names:
            top_class = results.names[int(results.boxes.cls[0])]
            confidence = results.boxes.conf[0].item()
            timestamp = frame_count / frame_rate
            if top_class not in class_durations:
                class_durations[top_class] = {'start': timestamp, 'duration': 0}
            else:
                class_durations[top_class]['duration'] += 1 / frame_rate

    cap.release()
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
    for class_name, times in durations.items():
        st.write(f"{class_name}: {times['duration']:.2f} segundos")
