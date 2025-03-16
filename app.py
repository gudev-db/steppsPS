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

        # Acessando os resultados de classificação
        if results.pandas().xywh[0].shape[0] > 0:  # Verificando se há resultados
            class_probabilities = results.pandas().xywh[0]  # Pega as probabilidades em formato pandas

            for _, row in class_probabilities.iterrows():
                class_name = row['name']  # Nome da classe
                confidence = row['confidence']  # Confiança da classe

                if confidence > 0.1:  # Ignora classes com baixa confiança
                    timestamp = frame_count / frame_rate  # Calcula o timestamp baseado no número de quadros

                    # Armazenar a duração de cada classe
                    if class_name not in class_durations:
                        class_durations[class_name] = {'start': timestamp, 'duration': 0}
                    class_durations[class_name]['duration'] += 1 / frame_rate  # Incrementa a duração

                    # Adicionar o nome da classe no frame
                    cv2.putText(frame, f"{class_name}: {confidence*100:.2f}%", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Exibir o frame no Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    os.remove(temp_file_path)

    # Exibir a duração de cada classe após o processamento
    st.write("Duração de cada classe detectada:")
    for class_name, data in class_durations.items():
        st.write(f"{class_name}: {data['duration']:.2f} segundos")

# Interface no Streamlit
st.title("Classificação de Ações em Vídeos")
uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    durations = process_video(uploaded_file)
