# Usa una imagen base con soporte para GPU
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Instala dependencias esenciales
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Configura el entorno de trabajo
WORKDIR /app

# Clona el repositorio y entra en el directorio
RUN git clone https://github.com/nazarenoventrelli/OmniParser.git && \
    cd OmniParser && git lfs install && git lfs pull

# Instala dependencias
WORKDIR /app/OmniParser
RUN pip install --no-cache-dir -r requirements.txt

# Asegurar que OpenCV usa solo la versión sin GUI
RUN pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless

# Descarga los pesos del modelo correctamente
RUN mkdir -p weights/icon_detect weights/icon_caption_florence
RUN git clone https://huggingface.co/microsoft/OmniParser-v2.0 weights/huggingface_models
RUN mv weights/huggingface_models/icon_detect/* weights/icon_detect/
RUN mv weights/huggingface_models/icon_caption/* weights/icon_caption_florence/
RUN rm -rf weights/huggingface_models

# Cambiar el directorio de trabajo a omnitool
WORKDIR /app/OmniParser/omnitool

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la API
CMD ["python3", "omniparserserver/omniparserserver.py", "--som_model_path", "../weights/icon_detect/model.pt", "--caption_model_name", "florence2", "--caption_model_path", "../weights/icon_caption_florence", "--device", "cuda", "--BOX_TRESHOLD", "0.05"]
