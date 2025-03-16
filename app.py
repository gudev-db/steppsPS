import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os

def load_model(model_path):
    model = YOLO(model_path)  # Carregar o modelo
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

        # Classificar o quadro atual
        results = model(frame)  # Inferência no quadro

        if results:
            # A estrutura de resultados pode variar, então vamos imprimir para entender
            print(results)

            # Para classificação, verificamos o índice da classe com maior confiança
            predicted_class = results[0].names[int(results[0].cls[0].item())]  # Classe predita
            confidence_score = results[0].conf[0].item()  # Confiança da classe

            # Anotando o quadro com a classe predita
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"Classe: {predicted_class} ({confidence_score:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Atualizando o vídeo no Streamlit
            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

            current_time = frame_count / fps

            # Atualizando a duração da classe detectada
            if predicted_class != previous_class:
                if previous_class is not None:
                    class_appearances[previous_class].append([start_time, current_time])
                start_time = current_time
            previous_class = predicted_class

        frame_count += 1

    # Finalizando a última duração
    if previous_class is not None:
        class_appearances[previous_class].append([start_time, current_time])

    cap.release()

    # Calculando a duração total para cada classe
    class_durations = {}
    for class_name, intervals in class_appearances.items():
        total_duration = sum(end - start for start, end in intervals)
        class_durations[class_name] = total_duration

    return class_durations

def main():
    st.title("Classificação de Classes em Vídeo com YOLO")

    # Caminho do modelo
    model_path = "classW.pt"  # Modelo de classificação YOLOv8
    model = load_model(model_path)
    st.success("Modelo carregado com sucesso!")

    video_file = st.file_uploader("Carregue um vídeo MP4", type=["mp4"])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.getbuffer())
            video_path = tmp_video.name

        # Exibindo o vídeo
        st.video(video_path)

        if st.button("Processar Vídeo"):
            with st.spinner("Processando vídeo..."):
                class_durations = process_video(model, video_path)
                st.success("Processamento concluído!")

                # Exibindo as durações de cada classe
                st.subheader("Duração Total de Cada Classe:")
                for class_name, duration in class_durations.items():
                    st.write(f"Classe {class_name}: {duration:.2f} segundos")

            os.unlink(video_path)

if __name__ == "__main__":
    main()
