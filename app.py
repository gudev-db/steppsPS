import cv2
import time
import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# Carregar o modelo YOLO treinado
model = YOLO('classW.pt')  # Substitua pelo caminho correto para o seu modelo treinado

# Função para processar o vídeo e detectar as ações
def process_video(video_file):
    # Criar um arquivo temporário para o vídeo
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name
    
    # Abrir o vídeo usando o OpenCV
    cap = cv2.VideoCapture(temp_file_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Taxa de quadros do vídeo
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / frame_rate  # Duração total do vídeo
    
    action_times = []  # Lista para armazenar as ações detectadas e seus tempos
    action_durations = []  # Lista para armazenar as durações das ações
    current_action = None
    start_time = None
    action_durations_dict = {  # Dicionário para armazenar o tempo de cada ação
        'ApplyEyeMakeup': 0,
        'ApplyLipstick': 0,
        'BlowDryHair': 0,
        'BrushingTeeth': 0,
        'Haircut': 0
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar a classe da ação no quadro atual
        results = model(frame)  # Detecção no frame (classificação)
        
        # Obter a classe com maior probabilidade
        probs = results.probs[0]  # Probabilidades para o primeiro (e único) quadro
        max_prob_index = probs.argmax()
        predicted_class = results.names[max_prob_index]  # Nome da classe com maior probabilidade
        
        # Verificar se houve uma mudança na ação
        if predicted_class != current_action:
            if current_action is not None and start_time is not None:
                # Armazenar a duração da ação anterior
                action_durations_dict[current_action] += time.time() - start_time
            
            # Atualizar a ação atual
            current_action = predicted_class
            start_time = time.time()  # Iniciar o tempo da nova ação
        
        time.sleep(1 / frame_rate)  # Atraso para simular a taxa de quadros
    
    # Adicionar a última ação detectada
    if current_action is not None and start_time is not None:
        action_durations_dict[current_action] += time.time() - start_time
    
    cap.release()
    
    # Limpar o arquivo temporário
    os.remove(temp_file_path)
    
    # Gerar o relatório de ações e durações
    action_report = {}
    for action, duration in action_durations_dict.items():
        if duration > 0:  # Apenas incluir ações com durações positivas
            action_report[action] = round(duration, 2)
    
    return action_report, action_durations_dict

# Interface Streamlit
st.title('Detecção de Ações Humanas em Vídeos')

# Carregar o vídeo
uploaded_video = st.file_uploader("Faça upload de um vídeo", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Exibir o vídeo carregado
    st.video(uploaded_video)
    
    # Processar o vídeo e obter as ações e durações
    action_report, action_durations_dict = process_video(uploaded_video)
    
    # Exibir as ações detectadas
    st.write("Ações Detectadas:")
    for action, duration in action_report.items():
        st.write(f"{action}: {duration} segundos")
    
    # Gerar um arquivo de relatório
    report_file = "action_report.txt"
    with open(report_file, "w") as f:
        for action, duration in action_report.items():
            f.write(f"{action}: {duration} segundos\n")
    
    # Botão para download do relatório
    st.download_button("Baixar Relatório", report_file)
