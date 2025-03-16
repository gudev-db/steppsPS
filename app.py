import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import tempfile

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
        results = model(frame)
        
        # Extrair os resultados de classificação
        if results.names:
            # Obtém a classe com maior probabilidade
            class_probs = results.probs[0].cpu().numpy()
            max_prob_class_idx = np.argmax(class_probs)
            class_name = results.names[max_prob_class_idx]
            class_prob = class_probs[max_prob_class_idx]
            
            # Exibir o vídeo com a classificação
            cv2.putText(frame, f"{class_name} ({class_prob:.2f})", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Exibindo o frame no Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            # Atualizando o tempo de duração de cada classe
            if class_name in class_durations:
                class_durations[class_name]['duration'] += 1 / frame_rate
            else:
                class_durations[class_name] = {'duration': 1 / frame_rate, 'prob': class_prob}

        time.sleep(1 / frame_rate)  # Atraso para simular o FPS original do vídeo

    # Fechar o vídeo
    cap.release()

    # Exibir os resultados ao final
    st.write("Duração de cada classe (em segundos):")
    for class_name, data in class_durations.items():
        st.write(f"{class_name}: {data['duration']:.2f} segundos, Probabilidade média: {data['prob']:.2f}")

# Interface do Streamlit
st.title("Classificação de Ações no Vídeo")
uploaded_file = st.file_uploader("Carregue um vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    durations = process_video(uploaded_file)
