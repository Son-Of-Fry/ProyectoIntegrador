# camera_pose_qt.py
import sys, os, cv2, torch
from ultralytics import YOLO
from PySide6 import QtCore, QtGui, QtWidgets

# ===============================
# CONFIGURACIÓN
# ===============================
MODEL_PATH = "runs/pose/train/weights/best.pt"   # ✅ Ruta al modelo correcto
CAM_INDEX = 2
CONF_THRESHOLD = 0.5
WINDOW_TITLE = "Human Activity Detection - YOLOv11 Pose"

# ===============================
# CLASE PRINCIPAL (Qt)
# ===============================
class PoseViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(960, 720)

        # --- UI ---
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)

        # --- Cargar modelo ---
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ No se encontró el modelo en: {MODEL_PATH}")

        self.model = YOLO(MODEL_PATH)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Modelo cargado en {self.device.upper()}")

        # --- Abrir cámara ---
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("❌ No se pudo abrir la cámara.")

        print("🎥 Cámara iniciada. Cierra la ventana o presiona Ctrl+C para salir.")

        # --- Timer de actualización ---
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms → ~33 FPS

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        # --- Inferencia YOLO Pose ---
        results = self.model.predict(frame, conf=CONF_THRESHOLD, device=self.device, verbose=False)
        annotated = results[0].plot()  # Dibuja keypoints y etiquetas

        # --- Mostrar clase detectada (opcionalmente grande arriba del frame) ---
        if len(results[0].boxes.cls) > 0:
            cls_id = int(results[0].boxes.cls[0])
            class_name = self.model.names.get(cls_id, "Desconocido")
            cv2.putText(
                annotated, f"Actividad: {class_name}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
            )

        # --- Convertir a formato Qt ---
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        print("🛑 Cerrando cámara...")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = PoseViewer()
    viewer.show()
    sys.exit(app.exec())
