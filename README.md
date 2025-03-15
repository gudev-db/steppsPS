# steppsPS

Detecção de Classes em Vídeo com YOLOv11
Este projeto é uma aplicação Streamlit que permite carregar um vídeo MP4, processá-lo usando um modelo YOLOv11 pré-treinado e exibir a duração total (em segundos) que cada classe aparece no vídeo. O modelo é carregado localmente a partir de um arquivo .pt.

Requisitos
Antes de executar o aplicativo, certifique-se de ter instalado as seguintes bibliotecas:

bash
Copy
pip install streamlit ultralytics opencv-python
Como Usar
Coloque o Modelo na Pasta:

Certifique-se de que o arquivo do modelo YOLOv11 (por exemplo, yolo11n.pt) esteja na mesma pasta do código.

Execute o Aplicativo:

No terminal, navegue até a pasta do projeto e execute o seguinte comando:

bash
Copy
streamlit run app_yolov11.py
Carregue um Vídeo:

Na interface do Streamlit, carregue um vídeo no formato MP4.

Processe o Vídeo:

Clique no botão "Processar Vídeo" para iniciar a detecção de classes.

O vídeo será exibido com as detecções em tempo real.

Após o processamento, a duração total de cada classe será exibida.

Exemplo de Saída
Após o processamento, o aplicativo exibirá algo como:

Copy
Duração Total de Cada Classe:
Classe pessoa: 12.34 segundos
Classe carro: 8.76 segundos
Estrutura do Projeto
app_yolov11.py: Código principal do aplicativo Streamlit.

yolo11n.pt: Arquivo do modelo YOLOv11 (deve estar na mesma pasta do código).

Observações
O aplicativo processa o vídeo em tempo real, o que pode ser lento para vídeos longos. Para melhor desempenho, use uma GPU.

O aplicativo suporta vídeos no formato MP4. Para outros formatos, ajuste o código.
