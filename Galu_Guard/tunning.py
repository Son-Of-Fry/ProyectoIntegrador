from ultralytics import YOLO

# Carga el modelo YOLOv11s preentrenado
model = YOLO('yolo11s.pt')  # nombre correcto del archivo
# Entrena con tu dataset
model.train(
    data='training/t1/data.yaml',
    epochs=50,
    imgsz=640,
    device=0  # Usa la GPU si está disponible
)
