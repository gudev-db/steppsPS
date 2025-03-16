![image](https://github.com/user-attachments/assets/62a9c8c5-78a3-4a44-8dd7-fb136a4b7252)O aplicativo está hospedado e rodando e pode ser acessado por aqui:
https://stepps-ps-gustavo-romao.streamlit.app/

![Exemplo de uso](imagem.png)

# Detecção de Classes em Vídeo com YOLOv11 e Streamlit
Este aplicativo Streamlit permite que você faça upload de um vídeo MP4 e de um arquivo de modelo YOLOv11 (.pt). Ele processa o vídeo para detectar várias classes (objetos) usando o modelo YOLOv11, exibe os quadros anotados e calcula a duração total de cada classe detectada no vídeo.

## Funcionalidades:
Faça o upload de um arquivo de vídeo MP4 para detecção de classes.
Exiba os quadros anotados com caixas delimitadoras e rótulos de classe em tempo real.
Calcule e exiba a duração total de cada classe detectada no vídeo.


## Requisitos:
Python 3.x
Streamlit
OpenCV
PyTorch
Ultralytics YOLOv11
Instalação
Clone este repositório:


git clone <url-do-repositório>
cd <diretório-do-repositório>
Instale as dependências necessárias:

### Instale as dependências usando pip:
pip install streamlit opencv-python ultralytics

### Baixe o modelo YOLOv11:

Certifique-se de ter um arquivo de modelo YOLOv11 (.pt). Este arquivo de modelo deve ser compatível com o conteúdo do vídeo que você deseja processar.
Execute o aplicativo:

### Inicie o aplicativo Streamlit com o seguinte comando:

streamlit run app.py

### Como Usar:
### Faça o upload do Modelo YOLOv11:

Certifique-se de que seu arquivo de modelo está no mesmo diretório que o aplicativo ou forneça seu caminho.
O aplicativo tentará carregar o modelo do caminho especificado (model_path), sendo o caminho padrão finalW.pt.
Faça o upload do Vídeo:

Clique no botão "Carregue um vídeo MP4" e selecione um arquivo de vídeo MP4 do seu computador.
Processar o Vídeo:

Após o upload do vídeo, clique no botão "Processar Vídeo" para começar a detecção de classes no vídeo.
