import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os

def load_model(model_path):
    model = YOLO(model_path)
    return model

def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    class_appearances = defaultdict(list)

    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        current_time = frame_count / fps
        highest_confidence_detection = None

        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                if highest_confidence_detection is None or confidence > highest_confidence_detection[1]:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    highest_confidence_detection = (class_name, confidence)

        if highest_confidence_detection:
            class_name = highest_confidence_detection[0]
            if not class_appearances[class_name] or class_appearances[class_name][-1][1] < current_time:
                class_appearances[class_name].append([current_time, current_time + 1 / fps])
            else:
                class_appearances[class_name][-1][1] = current_time + 1 / fps

        frame_count += 1

    cap.release()

    class_durations = {}
    for class_name, intervals in class_appearances.items():
        total_duration = sum(end - start for start, end in intervals)
        class_durations[class_name] = total_duration

    return class_durations

def main():
    st.title("Detecção de Classes em Vídeo com YOLOv11")

    model_path = "finalW.pt"  
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
