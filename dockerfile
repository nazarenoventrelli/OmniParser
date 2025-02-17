# Usa una imagen base con soporte para GPU si es necesario
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Instala dependencias esenciales
    RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1-mesa-glx && \
    apt-get clean

# Configura el entorno de trabajo
WORKDIR /app

# Clona el repositorio
RUN git clone https://github.com/nazarenoventrelli/OmniParser.git

# Entra en el repositorio
WORKDIR /app/OmniParser

# Instala dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copia los pesos del modelo a la carpeta adecuada (esto puede cambiar si los tienes en otro lado)
RUN mkdir -p weights/icon_detect weights/icon_caption_florence
RUN huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "icon_detect/*"
RUN huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "icon_caption/*"
RUN mv weights/icon_caption weights/icon_caption_florence

# Cambiar el directorio de trabajo a omnitool (donde está omniparserserver.py)
WORKDIR /app/OmniParser/omnitool

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la API
CMD ["python3", "omniparserserver/omniparserserver.py", "--som_model_path", "../weights/icon_detect/model.pt", "--caption_model_name", "florence2", "--caption_model_path", "../weights/icon_caption_florence", "--device", "cuda", "--BOX_TRESHOLD", "0.05"]
