import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os

def load_model(model_path):
    """
    Load the custom YOLO classification model from the given path.
    """
    model = YOLO(model_path)  # Load your custom model
    return model

def process_video(model, video_path):
    """
    Process the video and track the duration of each class detected.
    """
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

        # Classify the frame
        results = model(frame)  # Model inference on the current frame

        if results:
            # Get the predicted class for the first box (if exists)
            predicted_class = results[0].names[int(results[0].boxes.cls[0].item())]
            
            # Annotate the frame with the predicted class
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"Classe: {predicted_class}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Update the Streamlit video placeholder with the annotated frame
            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

            current_time = frame_count / fps

            # Track the duration of consecutive frames with the same class
            if predicted_class != previous_class:
                if previous_class is not None:
                    class_appearances[previous_class].append([start_time, current_time])
                start_time = current_time
            previous_class = predicted_class

        frame_count += 1

    # Add the last segment of video class tracking
    if previous_class is not None:
        class_appearances[previous_class].append([start_time, current_time])

    cap.release()

    # Calculate the total duration for each class
    class_durations = {}
    for class_name, intervals in class_appearances.items():
        total_duration = sum(end - start for start, end in intervals)
        class_durations[class_name] = total_duration

    return class_durations

def main():
    """
    Main function to run the Streamlit app for video classification.
    """
    st.title("Classificação de Classes em Vídeo com YOLO")

    # Define the path to your custom model
    model_path = "classW.pt"  # Replace with the path to your custom classification model
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado no caminho: {model_path}")
        return

    model = load_model(model_path)
    st.success("Modelo carregado com sucesso!")

    # Upload video
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

                # Display the class durations
                st.subheader("Duração Total de Cada Classe:")
                for class_name, duration in class_durations.items():
                    st.write(f"Classe {class_name}: {duration:.2f} segundos")

            os.unlink(video_path)  # Remove the temporary video file after processing

if __name__ == "__main__":
    main()
