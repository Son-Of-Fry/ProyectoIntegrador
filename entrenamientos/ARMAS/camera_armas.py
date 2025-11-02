# camera_epp_qt.py
import sys, os, cv2, torch
from ultralytics import YOLO
from PySide6 import QtCore, QtGui, QtWidgets

# ===============================
# CONFIGURACIÓN
# ===============================
MODEL_PATH = "GUN_best.pt"   # Ruta al modelo entrenado
CAM_INDEX = 0               # Índice de cámara (0 = webcam principal)
CONF_THRESHOLD = 0.80
WINDOW_TITLE = "EPP Detection (Qt GUI)"

# ===============================
# CLASE PRINCIPAL (QT)
# ===============================
class EPPViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(960, 720)

        # Elementos UI
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)

        # Carga modelo YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Modelo cargado en {self.device.upper()}")

        # Abre cámara
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("❌ No se pudo abrir la cámara.")

        print("🎥 Cámara iniciada. Presiona Ctrl+C o cierra la ventana para salir.")

        # Timer para actualizar frames
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # cada 30 ms ≈ 33 FPS

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        # Inferencia YOLO
        results = self.model(frame, conf=CONF_THRESHOLD, device=self.device, verbose=False)
        annotated = results[0].plot()

        # Convertir a formato Qt
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        # Mostrar en el QLabel
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        print("🛑 Cerrando cámara y liberando recursos...")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = EPPViewer()
    viewer.show()
    sys.exit(app.exec())
