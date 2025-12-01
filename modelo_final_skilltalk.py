# -*- coding: utf-8 -*-
"""
Script Principal: Clasificación de Postura Corporal con Mediapipe, Normalización
y Modelo MLP-LSTM. Adaptado para despliegue con Gradio.

OPTIMIZACIÓN DE MEMORIA CLAVE: Los frames BGR no se almacenan en RAM.
El video se re-lee en la etapa de visualización para evitar el error OOM.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Optional, Dict, Tuple, Callable
from collections import Counter
import os 
# Importamos Gradio para el tipo de objeto Progress, asumiendo gr se importa en app.py
try:
    import gradio as gr
except ImportError:
    # Definición mínima para que el script pueda correr sin Gradio instalado
    class DummyProgress:
        def __call__(self, *args, **kwargs):
            pass
    gr = None

# ====================================================================
## ⚙️ PARÁMETROS DE CONFIGURACIÓN
# ====================================================================

# 🛑 1. RUTAS Y ARCHIVOS
MODEL_PATH = "mlp_lstm_ted_final.h5"

# 🛑 2. PARÁMETROS DEL MODELO Y PROCESAMIENTO
CHUNK_SIZE = 30 
CLASS_NAMES = ["Beat", "No-Gesture"] 
COLORS = {
    "Beat": (0, 255, 0),    
    "No-Gesture": (255, 0, 0) 
}

# ⚡ OPTIMIZACIÓN CLAVE (PARA VELOCIDAD Y MEMORIA): 
# Factor de Salto de Fotogramas (Frames to Skip). 
# Aumentado a 15 para un procesamiento mucho más rápido y menor uso de memoria.
FRAME_SKIP_FACTOR = 5 

# 🛑 3. CONSTANTES DEL ESQUELETO (Kinect v2)
SPINE_BASE = 0; SPINE_MID = 1; NECK = 2; HEAD = 3
SHOULDER_LEFT = 4; ELBOW_LEFT = 5; WRIST_LEFT = 6; HAND_LEFT = 7
SHOULDER_RIGHT = 8; ELBOW_RIGHT = 9; WRIST_RIGHT = 10; HAND_RIGHT = 11
HIP_LEFT = 12; KNEE_LEFT = 13; ANKLE_LEFT = 14; FOOT_LEFT = 15
HIP_RIGHT = 16; KNEE_RIGHT = 17; ANKLE_RIGHT = 18; FOOT_RIGHT = 19
SPINE_SHOULDER = 20; HANDTIP_LEFT = 21; THUMB_LEFT = 22
HANDTIP_RIGHT = 23; THUMB_RIGHT = 24

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ====================================================================
## 📏 FUNCIONES DE PREPROCESAMIENTO (SIN CAMBIOS)
# ====================================================================

def normalize_skeleton_sequence(seq: np.ndarray) -> np.ndarray:
    """Normaliza una secuencia completa de esqueletos (T, 25, 3)."""
    seq = seq.copy().astype(np.float32)
    seq[np.isnan(seq)] = 0.0

    # 1. Centrar en pelvis (SPINE_BASE)
    root = seq[:, SPINE_BASE:SPINE_BASE+1, :]
    seq = seq - root

    # 2. Rotación para alinear hombros con eje X
    left_shoulder = seq[:, SHOULDER_LEFT, :]
    right_shoulder = seq[:, SHOULDER_RIGHT, :]
    shoulder_vec = np.mean(left_shoulder - right_shoulder, axis=0)
    shoulder_vec[1] = 0

    norm = np.linalg.norm(shoulder_vec)
    if norm < 1e-6:
        shoulder_vec = np.array([1.0, 0.0, 0.0])
    else:
        shoulder_vec = shoulder_vec / norm

    target = np.array([1.0, 0.0, 0.0])
    v = np.cross(shoulder_vec, target)
    c = np.dot(shoulder_vec, target)

    if np.linalg.norm(v) < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    seq = seq @ R.T

    # 3. Escalar por distancia entre hombros
    shoulder_dist = np.mean(np.linalg.norm(seq[:, SHOULDER_LEFT, :] - seq[:, SHOULDER_RIGHT, :], axis=1))
    scale = 1.0 / shoulder_dist if shoulder_dist > 1e-6 else 1.0
    seq = seq * scale

    # 4. Normalización de longitud de huesos
    bones = [(SPINE_BASE, SPINE_MID), (SPINE_MID, SPINE_SHOULDER), (SPINE_SHOULDER, NECK), (NECK, HEAD),
             (SPINE_SHOULDER, SHOULDER_LEFT), (SHOULDER_LEFT, ELBOW_LEFT), (ELBOW_LEFT, WRIST_LEFT), (WRIST_LEFT, HAND_LEFT),
             (SPINE_SHOULDER, SHOULDER_RIGHT), (SHOULDER_RIGHT, ELBOW_RIGHT), (ELBOW_RIGHT, WRIST_RIGHT), (WRIST_RIGHT, HAND_RIGHT),
             (SPINE_BASE, HIP_LEFT), (HIP_LEFT, KNEE_LEFT), (KNEE_LEFT, ANKLE_LEFT), (ANKLE_LEFT, FOOT_LEFT),
             (SPINE_BASE, HIP_RIGHT), (HIP_RIGHT, KNEE_RIGHT), (KNEE_RIGHT, ANKLE_RIGHT), (ANKLE_RIGHT, FOOT_RIGHT)]

    for j1, j2 in bones:
        vec = seq[:, j2] - seq[:, j1]
        avg_len = np.mean(np.linalg.norm(vec, axis=1))
        if avg_len > 1e-6:
            seq[:, j2] = seq[:, j1] + vec / avg_len

    return seq

# ... (Funciones auxiliares de mapeo y chunking, sin cambios)
## 📐 MAPEO DE LANDMARKS (MEDIAPIPE → KINECT25)
def compute_spine_points(landmarks):
    def to_np(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    left_hip = to_np(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = to_np(mp_pose.PoseLandmark.RIGHT_HIP)
    left_sh = to_np(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_sh = to_np(mp_pose.PoseLandmark.RIGHT_SHOULDER)

    spine_base = (left_hip + right_hip) / 2.0
    spine_shoulder = (left_sh + right_sh) / 2.0
    spine_mid = (spine_base + spine_shoulder) / 2.0
    return spine_base, spine_mid, spine_shoulder

def extract_kinect25_from_mediapipe(landmarks) -> np.ndarray:
    def L(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    spine_base, spine_mid, spine_shoulder = compute_spine_points(landmarks)

    k = np.zeros((25, 3), dtype=np.float32)

    k[0] = spine_base; k[1] = spine_mid; k[2] = spine_shoulder; k[3] = L(mp_pose.PoseLandmark.NOSE)
    k[4] = L(mp_pose.PoseLandmark.LEFT_SHOULDER); k[5] = L(mp_pose.PoseLandmark.LEFT_ELBOW); k[6] = L(mp_pose.PoseLandmark.LEFT_WRIST)
    k[7] = L(mp_pose.PoseLandmark.LEFT_INDEX)
    k[8] = L(mp_pose.PoseLandmark.RIGHT_SHOULDER); k[9] = L(mp_pose.PoseLandmark.RIGHT_ELBOW); k[10] = L(mp_pose.PoseLandmark.RIGHT_WRIST)
    k[11] = L(mp_pose.PoseLandmark.RIGHT_INDEX)
    k[12] = L(mp_pose.PoseLandmark.LEFT_HIP); k[13] = L(mp_pose.PoseLandmark.LEFT_KNEE); k[14] = L(mp_pose.PoseLandmark.LEFT_ANKLE); k[15] = L(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    k[16] = L(mp_pose.PoseLandmark.RIGHT_HIP); k[17] = L(mp_pose.PoseLandmark.RIGHT_KNEE); k[18] = L(mp_pose.PoseLandmark.RIGHT_ANKLE); k[19] = L(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    k[20] = spine_shoulder
    k[21] = L(mp_pose.PoseLandmark.LEFT_INDEX); k[22] = L(mp_pose.PoseLandmark.LEFT_THUMB); k[23] = L(mp_pose.PoseLandmark.RIGHT_INDEX); k[24] = L(mp_pose.PoseLandmark.RIGHT_THUMB)
    return k

## 📦 CHUNKING Y PREPARACIÓN DE ENTRADAS
def create_chunks_from_skeletons(skeletons: List[np.ndarray], chunk_size: int) -> np.ndarray:
    if len(skeletons) == 0:
        return np.zeros((0, chunk_size, 25, 3), dtype=np.float32)
    sk_arr = np.stack(skeletons, axis=0)
    T = sk_arr.shape[0]
    chunks = []
    for start in range(0, T, chunk_size):
        end = start + chunk_size
        chunk = sk_arr[start:end]
        if chunk.shape[0] < chunk_size:
            last = chunk[-1] if chunk.shape[0] > 0 else np.zeros((25,3), dtype=np.float32)
            pad = np.repeat(last[None, :, :], chunk_size - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunks.append(chunk)
    return np.stack(chunks, axis=0).astype(np.float32)

def prepare_chunks_for_model(chunks_4d: np.ndarray) -> np.ndarray:
    N, chunk_len, J, C = chunks_4d.shape
    return chunks_4d.reshape(N, chunk_len, J * C)
# ... (Fin de funciones auxiliares)

## 💾 PROCESAMIENTO DE VIDEO Y EXTRACCIÓN (OPTIMIZADO PARA MEMORIA)
def process_video_to_kinect25_light(video_path: str, repeat_last_valid: bool = True, progress: Optional[Callable] = None) -> List[Dict]:
    """
    Lee el video y extrae SOLO los datos ligeros del esqueleto (K25 y landmarks) 
    para ahorrar RAM.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ⚡ USANDO model_complexity=0 para mayor velocidad
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, 
                         enable_segmentation=False, min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)

    frame_data_light = []
    last_valid_k25 = None
    frame_count = 0
    REPORT_FREQUENCY = 100 

    print("→ Extrayendo Pose (SÓLO DATOS LIGEROS)...")

    while True:
        ret, frame_bgr = cap.read() 
        if not ret: break

        current_pose_landmarks = None
        current_k25 = np.zeros((25,3), dtype=np.float32)

        # 🛑 LÓGICA DE SALTO DE FOTOGRAMAS (MediaPipe solo en frames seleccionados)
        if frame_count % FRAME_SKIP_FACTOR == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) 
            res = pose.process(frame_rgb)

            if res.pose_landmarks:
                try:
                    current_k25 = extract_kinect25_from_mediapipe(res.pose_landmarks.landmark)
                    last_valid_k25 = current_k25.copy()
                    current_pose_landmarks = res.pose_landmarks
                except Exception:
                    if last_valid_k25 is not None and repeat_last_valid:
                        current_k25 = last_valid_k25.copy()
                    
            # Si MediaPipe no detecta nada y no hay last_valid_k25, current_k25 es np.zeros.
            # Si MediaPipe detecta pero falla la extracción, usamos last_valid_k25.

        # 🛑 LÓGICA PARA FOTOGRAMAS SALTADOS (Usamos la última pose válida)
        else:
            if last_valid_k25 is not None and repeat_last_valid:
                current_k25 = last_valid_k25.copy()
            # Los pose_landmarks son None, así no se dibujan en frames saltados.
        
        # Almacenar la información LIGERA
        frame_data_light.append({
            'k25': current_k25,
            # Guardamos los landmarks SÓLO si MediaPipe los generó (no es None).
            'pose_landmarks': current_pose_landmarks 
        })
        
        frame_count += 1
        
        # 🔔 REPORTE DE PROGRESO 🔔
        if progress and total_frames > 0 and frame_count % REPORT_FREQUENCY == 0:
            percentage = min(1.0, frame_count / total_frames)
            progress(percentage, desc=f"Paso 1/3: Extrayendo Pose: {frame_count}/{total_frames} frames procesados")
            
    cap.release()
    pose.close()
    return frame_data_light


## 🎨 DIBUJO Y ETIQUETADO POR CLASE (SIN CAMBIOS)
def draw_skeleton_and_label(image: np.ndarray, pose_landmarks, label: str, color: Tuple) -> np.ndarray:
    """Dibuja el esqueleto de MediaPipe y la etiqueta de clasificación."""
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        )
    text = f"CLASE: {label}"
    cv2.putText(image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return image


## 🎬 PIPELINE COMPLETO DE CLASIFICACIÓN Y VISUALIZACIÓN (MODIFICADO)
def classify_and_visualize_video(video_path: str, model_path: str, class_names: List[str], chunk_size: int, colors: Dict, progress: Optional[Callable] = None) -> List[np.ndarray]:
    
    # 1. Extracción y recolección de datos (solo datos ligeros)
    if progress:
        progress(0.05, desc="Paso 1/3: Iniciando Extracción de Pose")
        
    frame_data_light = process_video_to_kinect25_light(video_path, progress=progress)

    skeletons = [item['k25'] for item in frame_data_light]
    T = len(skeletons)
    if T == 0:
        raise RuntimeError("No se extrajeron esqueletos del video.")

    # 2. Chunking, Normalización y Predicción
    if progress:
        progress(0.70, desc="Paso 2/3: Chunking y Normalización de datos")
        
    chunks_4d = create_chunks_from_skeletons(skeletons, chunk_size=chunk_size)
    
    normalized_chunks = []
    for seq in chunks_4d:
        seq_norm = normalize_skeleton_sequence(seq)
        normalized_chunks.append(seq_norm)
    normalized_chunks = np.stack(normalized_chunks, axis=0)

    if progress:
        progress(0.85, desc="Paso 3/3: Predicción del modelo (MLP-LSTM)")
        
    X = prepare_chunks_for_model(normalized_chunks)
    model = load_model(model_path)
    preds = model.predict(X, verbose=0)
    pred_inds = preds.argmax(axis=1)

    # 3. Visualización: Re-leemos el video para dibujar (Ahorro de RAM)
    
    # Abrimos el video de entrada para leer los frames BGR originales
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo re-abrir el video: {video_path}")
        
    visual_frames = []
    frame_index = 0
    print("→ Dibujando y coloreando frames...")

    while True:
        ret, frame_bgr = cap.read() # <--- Leemos el frame BGR ORIGINAL
        if not ret: break

        if frame_index >= T: break # Por si acaso

        # Determinamos a qué chunk pertenece este frame
        chunk_idx = frame_index // chunk_size
        
        if chunk_idx < len(pred_inds):
            predicted_label = class_names[pred_inds[chunk_idx]]
            color = colors.get(predicted_label, (255, 255, 255))
        else:
             # Si hay un error en el conteo, usamos una etiqueta segura
            predicted_label = "Error" 
            color = (0, 0, 255) # Rojo

        # Obtenemos los landmarks almacenados (serán None en frames saltados)
        data = frame_data_light[frame_index]
        pose_landmarks_to_draw = data['pose_landmarks']

        visual_frame = draw_skeleton_and_label(frame_bgr.copy(), pose_landmarks_to_draw, predicted_label, color)
        visual_frames.append(visual_frame)

        frame_index += 1
    
    cap.release()
    return visual_frames

# ====================================================================
## 🎬 FUNCIÓN PRINCIPAL PARA GRADIO (Punto de entrada de la web)
# ====================================================================

def classify_and_save_feedback_video(input_video_path: str, output_video_path: str, progress=None) -> str:
    """
    Función adaptada para Gradio.
    """
    # Manejar el objeto progress si no se está ejecutando en Gradio
    if gr is None:
        progress = DummyProgress()
    
    print(f"\n--- INICIANDO PROCESAMIENTO ---")
    
    try:
        # 1. Ejecutar el pipeline (incluye reporte de progreso)
        visualized_frames = classify_and_visualize_video(
            input_video_path, MODEL_PATH, CLASS_NAMES, CHUNK_SIZE, COLORS, progress=progress
        )

        # 2. Escritura del video
        if visualized_frames:
            progress(0.95, desc="Finalizando: Guardando Video de Salida")
            
            H, W, _ = visualized_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

            cap = cv2.VideoCapture(input_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) 
            if fps <= 0: fps = 30 
            cap.release()

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

            # Escribir los frames
            for frame in visualized_frames:
                out.write(frame)

            out.release()
            progress(1.0, desc="✅ Proceso Finalizado")
            print(f"\n✅ Video de retroalimentación guardado en: {output_video_path}")
            return output_video_path
        else:
            raise RuntimeError("El pipeline no pudo generar frames de salida.")

    except RuntimeError as e:
        progress(1.0, desc="❌ Error de Ejecución")
        print(f"\n❌ Error de Ejecución: {e}")
        raise
    except Exception as e:
        progress(1.0, desc="❌ Error Inesperado")
        print(f"\n❌ Ocurrió un error inesperado: {e}")
        raise
