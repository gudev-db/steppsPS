import cv2
import time
import streamlit as st
from ultralytics import YOLO
import tempfile
import shutil

# Carregar o modelo treinado YOLO
model = YOLO('classW.pt')  # Substitua pelo caminho correto para o seu modelo

# Função para processar o vídeo e detectar ações
def process_video(video_file):
    # Criar um arquivo temporário para o vídeo
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name
    
    # Abrir o vídeo usando o OpenCV
    cap = cv2.VideoCapture(temp_file_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Taxa de quadros do vídeo
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / frame_rate  # Duração total do vídeo
    
    action_times = []  # Lista para armazenar os tempos das ações detectadas
    action_durations = []  # Lista para armazenar as durações de cada ação
    current_action = None
    start_time = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar ações no quadro atual
        results = model(frame)  # Detecção no frame
        predictions = results.pandas().xywh[0]  # Extrair os resultados
        
        # Verificar se a ação foi detectada
        if len(predictions) > 0:
            detected_action = predictions.iloc[0]['name']
            if detected_action != current_action:
                if current_action is not None and start_time is not None:
                    # Armazenar a ação anterior e sua duração
                    action_times.append((current_action, start_time))
                    action_durations.append(time.time() - start_time)
                
                # Atualizar a ação atual
                current_action = detected_action
                start_time = time.time()
        
        time.sleep(1 / frame_rate)  # Atraso para simular a taxa de quadros
    
    # Adicionar a última ação detectada
    if current_action is not None and start_time is not None:
        action_times.append((current_action, start_time))
        action_durations.append(time.time() - start_time)
    
    cap.release()
    
    # Limpar o arquivo temporário
    os.remove(temp_file_path)
    
    # Gerar o documento com as ações e durações
    action_report = {}
    for action, start in zip(action_times, action_durations):
        action_report[action[0]] = round(start, 2)
    
    return action_report, action_durations

# Interface Streamlit
st.title('Detecção de Ações Humanas em Vídeos')

# Carregar o vídeo
uploaded_video = st.file_uploader("Faça upload de um vídeo", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Exibir o vídeo carregado
    st.video(uploaded_video)
    
    # Processar o vídeo e obter as ações e durações
    action_report, action_durations = process_video(uploaded_video)
    
    # Exibir as ações detectadas
    st.write("Ações Detectadas:")
    for action, duration in zip(action_report.keys(), action_durations):
        st.write(f"{action}: {round(duration, 2)} segundos")
    
    # Gerar um arquivo de relatório
    report_file = "action_report.txt"
    with open(report_file, "w") as f:
        for action, duration in zip(action_report.keys(), action_durations):
            f.write(f"{action}: {round(duration, 2)} segundos\n")
    
    # Botão para download do relatório
    st.download_button("Baixar Relatório", report_file)
