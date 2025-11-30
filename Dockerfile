# 1. Usar una imagen base de Python 3.10
FROM python:3.10-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# 2. Instalar dependencias del sistema operativo (necesarias para OpenCV/MediaPipe)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Establecer el directorio de trabajo
WORKDIR /app

# 4. Copiar e instalar las dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar todo el código fuente del proyecto Y el modelo .h5
# (Esto copia app.py, modelo_final_skilltalk.py, y el modelo de 1.8MB)
COPY . .

# 6. Exponer el puerto por defecto de Gradio
EXPOSE 7860

# 7. Comando de ejecución: Iniciar la aplicación Gradio
CMD ["python", "app.py"]
