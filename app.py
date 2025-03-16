import streamlit as st
import cv2
import google.generativeai as genai
from PIL import Image
import io
import tempfile
import os
import time

# Configuração do Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Inicializa os modelos do Gemini
modelo_vision = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={
        "temperature": 0.1  # Ajuste a temperatura aqui
    }
)  # Modelo para imagens
modelo_texto = genai.GenerativeModel("gemini-1.5-flash")  # Modelo para texto

# Guias do cliente (feedbacks do cliente)
guias = """[Guia de comentários do cliente]"""

# Função para processar o vídeo e identificar ações com LLM
def processar_video_com_llm(uploaded_video):
    # Criar um arquivo temporário para o vídeo
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        temp_file_path = temp_file.name

    # Abrir o vídeo com OpenCV
    cap = cv2.VideoCapture(temp_file_path)
    
    action_report = []  # Lista para armazenar as ações detectadas
    
    # Definir as ações possíveis
    actions = ['ApplyEyeMakeup', 'ApplyLipstick', 'BlowDryHair', 'BrushingTeeth', 'Haircut']

    # Iterar sobre os quadros do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processar o quadro: gerar uma descrição (por exemplo, a partir de características da imagem)
        # Você pode integrar o modelo de detecção de objetos aqui ou gerar um resumo simples do quadro.
        # Aqui vamos usar o modelo LLM para descrever o que acontece no quadro.
        
        # Convertemos o quadro para uma imagem que pode ser usada na LLM
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        
        # Gerar uma descrição do quadro usando a LLM (modelo de visão)
        prompt = f"Descreva detalhadamente o que está acontecendo nesta imagem. As ações possíveis são: {', '.join(actions)}. A imagem é uma sequência de um vídeo."
        
        try:
            with st.spinner('Analisando o vídeo...'):
                resposta = modelo_vision.generate_content(
                    contents=[prompt, {"mime_type": "image/png", "data": img_bytes}]
                )
                descricao_imagem = resposta.text  # A descrição da imagem gerada pela LLM

                # Usamos a LLM para determinar a ação com base na descrição
                action_predicted = None
                for action in actions:
                    if action.lower() in descricao_imagem.lower():
                        action_predicted = action
                        break

                # Armazenar a ação detectada
                if action_predicted:
                    action_report.append(action_predicted)
                else:
                    action_report.append("Ação não detectada")

                # Exibe a descrição e a ação detectada
                st.write(f"Descrição do quadro: {descricao_imagem}")
                st.write(f"Ação detectada: {action_predicted}")

                # Opcional: Atrasar um pouco para simular a taxa de quadros
                time.sleep(1 / cap.get(cv2.CAP_PROP_FPS))
        
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o quadro: {e}")
            break

    cap.release()
    os.remove(temp_file_path)
    return action_report

# Função para exibir o vídeo e processar
def alinhar_video():
    st.subheader('Aprovação de Criativos (Vídeo)')

    # Criação de um estado para controlar o vídeo carregado
    if 'video' not in st.session_state:
        st.session_state.video = None

    # Upload do vídeo
    uploaded_video = st.file_uploader("Escolha um vídeo", type=["mp4", "avi"])
    if uploaded_video is not None:
        # Exibe o vídeo carregado
        st.video(uploaded_video)

        # Processa o vídeo e analisa as ações
        action_report = processar_video_com_llm(uploaded_video)

        # Exibe a lista de ações detectadas
        st.subheader('Ações Detectadas:')
        for action in action_report:
            st.write(f"- {action}")

    # Botão para remover o vídeo
    if st.button("Remover Vídeo"):
        st.session_state.video = None
        st.experimental_rerun()  # Atualiza a aplicação

    # Se um vídeo foi armazenado no estado da sessão, exibe a opção de remover
    if st.session_state.video is not None:
        st.info("Vídeo carregado. Clique no botão acima para removê-lo.")

# Chamar a função para exibir a interface de upload e análise de vídeo
alinhar_video()
