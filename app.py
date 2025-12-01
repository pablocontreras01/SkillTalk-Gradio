import gradio as gr
import os
import time
from typing import Optional

# 1. Importar la función y los parámetros clave desde tu script principal
from modelo_final_skilltalk import classify_and_save_feedback_video
from modelo_final_skilltalk import CHUNK_SIZE

# Directorio temporal para guardar videos procesados (necesario en despliegue web)
OUTPUT_DIR = "temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gradio_processor(video_path_input: Optional[str], progress=gr.Progress()) -> Optional[str]:
    """
    Función wrapper que Gradio llama al subir un archivo.
    Se asegura de pasar el objeto progress a la función principal.
    """
    if video_path_input is None:
        raise gr.Error("Por favor, sube un archivo de video para clasificar.")
        
    # Crear una ruta de salida temporal única para evitar conflictos
    timestamp = int(time.time())
    output_filename = f"feedback_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        # Llama a tu función adaptada
        # Pasamos el objeto progress de Gradio
        final_video_path = classify_and_save_feedback_video(video_path_input, output_path, progress=progress)
        
        return final_video_path
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        # gr.Error es la forma correcta de mostrar un mensaje de error en la UI de Gradio
        raise gr.Error(f"Error en el procesamiento del modelo: {e}")


# --- 2. Definición de la Interfaz de Gradio ---

iface = gr.Interface(
    # La función que Gradio debe ejecutar
    fn=gradio_processor,
    
    # ENTRADA: Un componente de video para subir archivos
    inputs=gr.Video(label="🎥 Sube el video del discurso"),
    
    # SALIDA: Un componente de video para mostrar el resultado
    outputs=gr.Video(label="✅ Video con Retroalimentación (Esqueleto Coloreado)"),
    
    title="🕺 Clasificador de Gestos Beat (SkillTalk)",
    description=f"El modelo clasifica los frames en 'Beat' (verde) o 'No-Gesture' (azul) usando una ventana deslizante de {CHUNK_SIZE} frames."
)

# 3. Iniciar la interfaz
# Se añade el parámetro server_timeout=300 (5 minutos)
iface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 7860)),
    server_timeout=300 # 💡 5 minutos = 300 segundos
)

# 3. Iniciar la interfaz
# server_name="0.0.0.0" permite el acceso externo (necesario para Docker/Render)
# server_port usa la variable de entorno $PORT (estándar para plataformas en la nube)
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
