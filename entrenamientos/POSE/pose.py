# ===========================================================
# HUMAN ACTIVITY RECOGNITION - YOLOv11 Pose (Python Script)
# ===========================================================
# Autor: Robotics Galu
# Descripción:
#   Entrena un modelo YOLOv11 pose (keypoints) para reconocer
#   actividades humanas a partir de un dataset de Roboflow.
#   Incluye entrenamiento, validación, inferencia en imágenes,
#   videos y cámara web en tiempo real.
# ===========================================================

import os
import glob
import torch
import cv2
from ultralytics import YOLO
from roboflow import Roboflow


# ===========================================================
# Step 01. Verificar entorno y GPU
# ===========================================================
print("=== Step 01. Verificar aceleración ===")

if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 GPU CUDA disponible: {torch.cuda.get_device_name(0)}")

elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    print("🍎 Aceleración MPS (Apple Neural Engine / Metal) disponible y activa.")

else:
    device = "cpu"
    print("⚙️ Usando CPU — el entrenamiento será más lento.")

print(f"✅ Dispositivo seleccionado automáticamente: {device}")


# ===========================================================
# Step 02. Descargar dataset de Roboflow
# ===========================================================
print("\n=== Step 02. Descargar dataset ===")

rf = Roboflow(api_key="hzoVrDHDbdeFNRLoh2Nq")
project = rf.workspace("pose-detection-twxbg").project("human-activity-ce7zu")
version = project.version(2)
dataset = version.download("yolov8")

DATASET_DIR = dataset.location
print("📁 Dataset descargado en:", DATASET_DIR)


# ===========================================================
# Step 03. Configurar modelo YOLOv11 pose
# ===========================================================
print("\n=== Step 03. Cargar modelo YOLOv11 pose ===")
model = YOLO("yolo11s-pose.pt")


# ===========================================================
# Step 04. Entrenar modelo
# ===========================================================
print("\n=== Step 04. Entrenar modelo ===")

results = model.train(
    data=f"{DATASET_DIR}/data.yaml",
    task="pose",
    mode="train",
    imgsz=640,
    epochs=50,
    batch=8,
    device=device
)
print("✅ Entrenamiento completado.")


# ===========================================================
# Step 05. Validar modelo
# ===========================================================
print("\n=== Step 05. Validar modelo ===")
metrics = model.val()
print("📊 Métricas:", metrics.results_dict)


# ===========================================================
# Step 06. Buscar el modelo best.pt más reciente
# ===========================================================
print("\n=== Step 06. Buscar modelo entrenado ===")

best_models = glob.glob("runs/pose/train*/weights/best.pt")
BEST = max(best_models, key=os.path.getmtime)
print("✅ Modelo más reciente:", BEST)


# ===========================================================
# Step 07. Resultados gráficos (opcional)
# ===========================================================
print("\n=== Step 07. Analizar resultados ===")
train_dir = os.path.dirname(os.path.dirname(BEST))
images_to_show = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PoseP_curve.png",
    "PoseR_curve.png",
    "val_batch1_pred.jpg",
    "val_batch2_pred.jpg"
]
for img in images_to_show:
    path = os.path.join(train_dir, img)
    if os.path.exists(path):
        print("🖼️", path)
    else:
        print("❌ No encontrado:", img)


# ===========================================================
# Step 08. Inferencia en imágenes de test
# ===========================================================
print("\n=== Step 08. Inferencia en imágenes de test ===")

test_path = os.path.join(DATASET_DIR, "test/images")
if os.path.exists(test_path):
    image_files = [os.path.join(test_path, f)
                   for f in os.listdir(test_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img in image_files:
        results = model.predict(source=img, save=True, conf=0.25)
    print("✅ Predicciones guardadas en runs/pose/predict/")
else:
    print("⚠️ No se encontró carpeta test/images.")


# ===========================================================
# Step 09. Mostrar resultados (opcional matplotlib)
# ===========================================================
print("\n=== Step 09. Mostrar predicciones ===")
pred_path = "runs/pose/predict"
if os.path.exists(pred_path):
    print("📸 Imágenes procesadas en:", pred_path)
else:
    print("⚠️ No se encontraron imágenes predichas.")


# ===========================================================
# Step 10. Inferencia en video
# ===========================================================
print("\n=== Step 10. Inferencia en video ===")

video_path = "video1.mp4"  # reemplaza con tu video local
if os.path.exists(video_path):
    model.predict(source=video_path, save=True, conf=0.25, iou=0.2)
    print("🎬 Video procesado y guardado en runs/pose/predict/")
else:
    print("⚠️ No se encontró video1.mp4, omitiendo este paso.")


# ===========================================================
# Step 11. Exportar modelo final
# ===========================================================
print("\n=== Step 11. Exportar modelo ===")

export_path = "HumanActivity_best.pt"
os.system(f'cp "{BEST}" "{export_path}"')
print(f"✅ Modelo exportado como: {export_path}")


# ===========================================================
# Step 12. Prueba en cámara web (tiempo real)
# ===========================================================
print("\n=== Step 12. Detección en cámara web ===")

if device != "cpu": torch.set_default_device(device)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
else:
    print("🎥 Cámara abierta. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Human Activity - YOLOv11 Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detección en cámara finalizada.")