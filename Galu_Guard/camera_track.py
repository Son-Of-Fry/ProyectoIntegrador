import sys, cv2, time, json, os
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

# ===================== CONFIGURACIÓN =====================
IS_WIN = sys.platform.startswith("win")
BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY

# Cargar configuración global
with open("config.json") as f:
    CONFIG = json.load(f)

# Parámetros YOLO desde config.json
MODEL_PATH = CONFIG["yolo"]["model"]
CONF = CONFIG["yolo"]["confidence"]
IMGSZ = CONFIG["yolo"]["imgsz"]
DEVICE = CONFIG["yolo"]["device"]
OBJECTS = set(CONFIG["objects"])

# ===================== HILO DE TRACKING =====================
class YOLOTrackerWorker(QtCore.QObject):
    """Hilo que ejecuta YOLO tracking sin bloquear la GUI."""
    result_ready = QtCore.Signal(object)
    process_request = QtCore.Signal(object)

    def __init__(self, model_path, tracker="botsort.yaml", conf=0.85, classes=[0], imgsz=640, device="cpu"):
        super().__init__()
        self.model = YOLO(model_path).to(device)
        self.tracker = tracker
        self.conf = conf
        self.classes = classes
        self.imgsz = imgsz
        self.device = device
        self.running = True
        self.process_request.connect(self.track_frame)

    @QtCore.Slot(object)
    def track_frame(self, frame):
        if not self.running:
            return
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            classes=self.classes,
            conf=self.conf,
            imgsz=self.imgsz,
            stream=False,
            verbose=False
        )
        annotated = results[0].plot()
        self.result_ready.emit(annotated)


# ===================== INTERFAZ PRINCIPAL =====================
class YoloTrackingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO11 Tracking (config.json)")
        self.resize(900, 600)

        # ---------- Layout ----------
        layout = QtWidgets.QVBoxLayout(self)
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # ---------- Fuente de cámara ----------
        cam_config = CONFIG.get("cameras", [{}])[0]
        cam_type = cam_config.get("type", "USB").upper()
        cam_id = cam_config.get("id", 0)

        if cam_type == "RTSP":
            print(f"🎥 Iniciando cámara RTSP: {cam_id}")
            self.cap = cv2.VideoCapture(cam_id)
        else:
            print(f"🎥 Iniciando cámara USB índice {cam_id}")
            self.cap = cv2.VideoCapture(int(cam_id), BACKEND)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("❌ No se pudo abrir la cámara definida en config.json.")

        # ---------- Hilo de tracking ----------
        self.worker = YOLOTrackerWorker(
            model_path=MODEL_PATH,
            tracker="botsort.yaml",
            conf=CONF,
            imgsz=IMGSZ,
            device=DEVICE,
            classes=[0]  # solo personas
        )
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.worker.result_ready.connect(self.update_image)
        self.thread.start()

        # ---------- Timer ----------
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(30)

        # ---------- FPS ----------
        self.label_fps = QtWidgets.QLabel("FPS: --", alignment=QtCore.Qt.AlignCenter)
        self.label_fps.setStyleSheet("font-size: 18px; color: white; background-color: #222; padding: 4px;")
        layout.addWidget(self.label_fps)

        self.last_time = time.time()

    # ===================== CAPTURA =====================
    def capture_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print("⚠️ No se pudo leer frame.")
            return

        now = time.time()
        fps = 1.0 / (now - self.last_time)
        self.last_time = now
        self.label_fps.setText(f"FPS: {fps:.1f}")

        self.worker.process_request.emit(frame)

    # ===================== ACTUALIZA INTERFAZ =====================
    def update_image(self, annotated_frame):
        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        )

    # ===================== LIMPIEZA =====================
    def closeEvent(self, e):
        self.timer.stop()
        self.worker.running = False
        self.thread.quit(); self.thread.wait()
        if self.cap:
            self.cap.release()
        e.accept()


# ===================== MAIN =====================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = YoloTrackingApp()
    w.show()
    sys.exit(app.exec())
