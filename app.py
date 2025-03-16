import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os

def load_model(model_path):
    # Carregar o modelo de classificação YOLO (certifique-se de que seja um modelo de classificação)
    model = YOLO(model_path)  # Substitua pelo caminho do seu modelo de classificação
    return model

def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Erro ao abrir o vídeo. Verifique o formato do arquivo.")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    class_appearances = defaultdict(list)
    previous_class = None
    start_time = 0

    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Classificar o quadro inteiro com o modelo YOLO
        results = model(frame)
        
        # Verifique se há alguma detecção e pegue a classe com maior probabilidade
        if results and results[0].pred[0].shape[0] > 0:
            # Obtendo a classe com maior probabilidade do quadro (modelo de classificação)
            predicted_class = results[0].names[int(results[0].pred[0][0].item())]  # Classe com maior probabilidade
            
            # Anotar o quadro com a classe prevista
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"Classe: {predicted_class}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

            current_time = frame_count / fps

            # Registrar a duração das sequências de quadros com a mesma classe
            if predicted_class != previous_class:
                if previous_class is not None:
                    class_appearances[previous_class].append([start_time, current_time])
                start_time = current_time
            previous_class = predicted_class

        frame_count += 1

    # Adicionar a última sequência
    if previous_class is not None:
        class_appearances[previous_class].append([start_time, current_time])

    cap.release()

    # Calcular a duração total de cada classe
    class_durations = {}
    for class_name, intervals in class_appearances.items():
        total_duration = sum(end - start for start, end in intervals)
        class_durations[class_name] = total_duration

    return class_durations

def main():
    st.title("Classificação de Classes em Vídeo com YOLO")

    model_path = "classW.pt"  # Substitua pelo caminho do seu modelo de classificação
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado no caminho: {model_path}")
        return

    model = load_model(model_path)
    st.success("Modelo carregado com sucesso!")

    video_file = st.file_uploader("Carregue um vídeo MP4", type=["mp4"])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.getbuffer())
            video_path = tmp_video.name

        st.video(video_path)

        if st.button("Processar Vídeo"):
            with st.spinner("Processando vídeo..."):
                class_durations = process_video(model, video_path)
                st.success("Processamento concluído!")

                st.subheader("Duração Total de Cada Classe:")
                for class_name, duration in class_durations.items():
                    st.write(f"Classe {class_name}: {duration:.2f} segundos")

            os.unlink(video_path)

if __name__ == "__main__":
    main()
