# ===========================================================
# YOLOv11 Entrenamiento y Detección de EPP (Python puro)
# ===========================================================
# Autor: Robotics Galu
# Descripción:
#   Entrena un modelo YOLOv11 para detección de equipos de protección personal
#   usando un dataset de Roboflow y Ultralytics.
#   Incluye validación, inferencia sobre imágenes, videos y cámara web.
# ===========================================================

import os
import glob
import torch
import cv2
from ultralytics import YOLO
from roboflow import Roboflow


# ===========================================================
# Step 01. Verificar GPU disponible
# ===========================================================
print("=== Step 01. Verificar aceleración ===")

device_train = None
device_infer = None

if torch.cuda.is_available():
    device_train = "cuda"
    device_infer = "cuda"
    print(f"GPU CUDA disponible: {torch.cuda.get_device_name(0)}")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device_infer = "mps"
    device_train = "mps"
    print("Advertencia: MPS está disponible pero tiene bugs conocidos en entrenamiento.")
    print("   Usando CPU para entrenamiento y MPS para inferencia.")
else:
    device_train = "cpu"
    device_infer = "cpu"
    print("Usando CPU — el entrenamiento será más lento.")

print(f"Dispositivo seleccionado para entrenamiento: {device_train}")
print(f"Dispositivo seleccionado para inferencia: {device_infer}")


# ===========================================================
# Step 02. Descargar dataset desde Roboflow
# ===========================================================
print("\n=== Step 02. Descargar dataset de Roboflow ===")

rf = Roboflow(api_key="hzoVrDHDbdeFNRLoh2Nq")
project = rf.workspace("project-uyrxf").project("ppe_detection-v1x3l")
version = project.version(2)
dataset = version.download("yolov11")

DATASET_DIR = dataset.location
print("Dataset descargado en:", DATASET_DIR)


# ===========================================================
# Step 03. Crear y entrenar el modelo YOLOv11n
# ===========================================================
print("\n=== Step 03. Entrenar modelo YOLOv11n ===")

model = YOLO("yolo11n.pt")

results = model.train(
    data=f"{DATASET_DIR}/data.yaml",
    epochs=80,
    imgsz=640,
    batch=-1,
    device=device_train,
    seed=42,
    patience=20,
    project="runs/detect",
    name="EPP_v11n"
)

print("Entrenamiento completado.")


# ===========================================================
# Step 04. Buscar el modelo best.pt más reciente
# ===========================================================
print("\n=== Step 04. Buscar modelo entrenado ===")

best_models = glob.glob("runs/detect/EPP_v11n*/weights/best.pt")
BEST = max(best_models, key=os.path.getmtime)
print("Modelo más reciente:", BEST)


# ===========================================================
# Step 05. Validar el modelo
# ===========================================================
print("\n=== Step 05. Validar modelo ===")
model = YOLO(BEST)
metrics = model.val(data=f"{DATASET_DIR}/data.yaml")
print("Resultados de validación:", metrics.results_dict)


# ===========================================================
# Step 06. Visualizar métricas (si existen archivos)
# ===========================================================
print("\n=== Step 06. Visualizar resultados ===")

train_dir = os.path.dirname(os.path.dirname(BEST))
images_to_show = [
    "results.png",
    "P_curve.png",
    "R_curve.png",
    "confusion_matrix_normalized.png",
]

for img_name in images_to_show:
    path = os.path.join(train_dir, img_name)
    if os.path.exists(path):
        print(" ", path)
    else:
        print("No encontrado:", img_name)


# ===========================================================
# Step 07. Inferencia sobre imágenes de prueba
# ===========================================================
print("\n=== Step 07. Inferencia en imágenes del dataset ===")

model.predict(
    source=f"{DATASET_DIR}/test/images",
    conf=0.25,
    save=True,
    project="runs/predict",
    name="EPP_images",
    device=device_infer
)
print("Inferencia en imágenes completada.")


# ===========================================================
# Step 08. Inferencia sobre video (si existe)
# ===========================================================
print("\n=== Step 08. Inferencia en video ===")

video_path = "PPE_demo.mp4"  # cambia este nombre por tu video local
if os.path.exists(video_path):
    model.predict(
        source=video_path,
        conf=0.25,
        save=True,
        project="runs/predict",
        name="EPP_video",
        device=device_infer
    )
    print("Video procesado y guardado en runs/predict/EPP_video")
else:
    print("No se encontró PPE_demo.mp4, omitiendo prueba de video.")


# ===========================================================
# Step 09. Exportar modelo final
# ===========================================================
print("\n=== Step 09. Exportar modelo ===")

export_path = "EPP_best.pt"
os.system(f'cp "{BEST}" "{export_path}"')
print(f"Modelo exportado a: {export_path}")


# ===========================================================
# Step 10. Comparar con modelo previo (opcional)
# ===========================================================
# Descomenta si deseas validar otro modelo preexistente.
"""
print("\n=== Step 10. Comparar con modelo previo ===")
prev_model = "best_prev.pt"
if os.path.exists(prev_model):
    prev = YOLO(prev_model)
    prev.val(data=f"{DATASET_DIR}/data.yaml")
else:
    print("No se encontró modelo previo para comparar.")
"""