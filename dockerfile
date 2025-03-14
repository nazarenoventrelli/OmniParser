# Usa una imagen base con soporte para GPU
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Instala dependencias esenciales
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 git-lfs \
    libgomp1 libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

# Configura el entorno de trabajo
WORKDIR /app

# Clona el repositorio y entra en el directorio
RUN git clone https://github.com/nazarenoventrelli/OmniParser.git && \
    cd OmniParser && git lfs install && git lfs pull

# Instala dependencias
WORKDIR /app/OmniParser
RUN pip install --no-cache-dir -r requirements.txt

# Instalar OpenCV completo con todas las dependencias necesarias
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Asegurar que OpenCV está correctamente instalado con todas las funcionalidades
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python==4.8.1.78

# Parchar el archivo utils.py de supervision para manejar el error de FONT_HERSHEY_SIMPLEX
RUN mkdir -p /tmp/patch && \
    echo 'import os\n\
import sys\n\
import supervision\n\
utils_path = os.path.join(os.path.dirname(supervision.__file__), "draw/utils.py")\n\
with open(utils_path, "r") as f:\n\
    content = f.read()\n\
content = content.replace("text_font: int = cv2.FONT_HERSHEY_SIMPLEX,", "text_font: int = 0,  # FONT_HERSHEY_SIMPLEX")\n\
with open(utils_path, "w") as f:\n\
    f.write(content)\n\
print("Supervision utils.py patched successfully!")' > /tmp/patch/patch_supervision.py && \
    python3 /tmp/patch/patch_supervision.py && \
    rm -rf /tmp/patch

# Configurar PaddlePaddle correctamente para CUDA
RUN pip uninstall -y paddlepaddle-gpu paddlepaddle && \
    pip install --no-cache-dir paddlepaddle-gpu==2.5.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

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
