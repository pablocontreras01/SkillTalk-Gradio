# 1. Usar una imagen base de Python compatible (Python 3.10 es una buena base para estas librerías)
FROM python:3.10-slim

# Evitar prompts interactivos durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# 2. Instalar dependencias del sistema operativo para OpenCV y MediaPipe
# (libglib2.0-0, libsm6, libxrender1, etc., son requeridas por OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Incluir build essentials por si acaso (gcc/g++)
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Establecer el directorio de trabajo
WORKDIR /app

# 4. Copiar e instalar las dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el código fuente del proyecto
# Esto copiará app.py, modelo_final_skilltalk.py, y el modelo .h5
COPY . .

# 6. Exponer el puerto por defecto de Gradio
# Gradio usa 7860 por defecto si no se especifica $PORT
EXPOSE 7860

# 7. Comando de ejecución: Iniciar la aplicación Gradio
# Gradio detectará automáticamente la variable de entorno $PORT proporcionada por Render
CMD ["python", "app.py"]
