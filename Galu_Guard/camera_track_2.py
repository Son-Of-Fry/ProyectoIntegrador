import sys, cv2, time, json, os, csv
from datetime import datetime
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO
from pathlib import Path

# ===================== CONFIGURACIÓN =====================
IS_WIN = sys.platform.startswith("win")
BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY

with open("config.json") as f:
    CONFIG = json.load(f)

MODEL_PATH = CONFIG["yolo"]["model"]
CONF = CONFIG["yolo"]["confidence"]
IMGSZ = CONFIG["yolo"]["imgsz"]
DEVICE = CONFIG["yolo"]["device"]
OBJECTS = set(CONFIG["objects"])

# Carpeta para reportes
REPORT_DIR = "tracking_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def _tracker_path(name: str) -> str:
    import ultralytics
    root = Path(ultralytics.__file__).resolve().parent
    p = root / "cfg" / "trackers" / name
    return str(p)

# ===================== WORKER =====================
class YOLOTrackerWorker(QtCore.QObject):
    frame_ready = QtCore.Signal(object, list)
    process_request = QtCore.Signal(object)

    def __init__(self, model_path, tracker="bytetrack.yaml", conf=0.85, classes=[0], imgsz=640, device="cpu"):
        super().__init__()
        self.model = YOLO(model_path).to(device)
        self.tracker = _tracker_path(tracker)
        self.conf = conf
        self.classes = classes
        self.imgsz = imgsz
        self.device = device
        self.running = True
        self.process_request.connect(self.track_frame)

        if not os.path.exists(self.tracker):
            try_botsort = _tracker_path("botsort.yaml")
            if os.path.exists(try_botsort):
                self.tracker = try_botsort

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

        boxes = results[0].boxes
        tracks = []
        if boxes.id is not None:
            now = datetime.now()
            fecha = now.strftime("%d/%m/%Y")
            hora = now.strftime("%H:%M:%S")
            for i, box in enumerate(boxes):
                track_id = int(box.id.tolist()[0])
                cls_id = int(box.cls.tolist()[0])
                conf = float(box.conf.tolist()[0])
                tracks.append({
                    "id": track_id,
                    "cls": results[0].names[cls_id],
                    "conf": conf,
                    "fecha": fecha,
                    "hora": hora
                })
        self.frame_ready.emit(annotated, tracks)

# ===================== DASHBOARD =====================
class YoloTrackingDashboard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galu Guard Tracking Dashboard")
        self.resize(1200, 700)

        main_layout = QtWidgets.QHBoxLayout(self)
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 7)

        # Tabla
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Clase", "Conf.", "Fecha", "Hora"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        left_layout.addWidget(QtWidgets.QLabel("🧾 Objetos rastreados", alignment=QtCore.Qt.AlignCenter))
        left_layout.addWidget(self.table)

        # Cámara
        cam_conf = CONFIG.get("cameras", [{}])[0]
        cam_type = cam_conf.get("type", "USB").upper()
        cam_id = cam_conf.get("id", 0)
        if cam_type == "RTSP":
            self.cap = cv2.VideoCapture(cam_id)
        else:
            self.cap = cv2.VideoCapture(int(cam_id), BACKEND)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("❌ No se pudo abrir la cámara.")

        # Video + info
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.video_label, stretch=10)

        self.info_label = QtWidgets.QLabel("FPS: -- | Tracks activos: 0 | Último reporte: --",
                                           alignment=QtCore.Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; color: white; background-color: #222; padding: 6px;")
        right_layout.addWidget(self.info_label)

        # Worker
        self.worker = YOLOTrackerWorker(MODEL_PATH, conf=CONF, imgsz=IMGSZ, device=DEVICE, classes=[0])
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.worker.frame_ready.connect(self.update_ui)
        self.thread.start()

        # Timers
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(30)

        # Guardado histórico cada 5 minutos
        self.csv_timer = QtCore.QTimer(self)
        self.csv_timer.timeout.connect(self.save_report)
        self.csv_timer.start(300000)  # 5 min

        self.report_data = []
        self.last_time = time.time()
        self.last_report_name = "--"

    def capture_frame(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.worker.process_request.emit(frame)

    def update_ui(self, frame, tracks):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

        for t in tracks:
            tid = str(t["id"])
            existing = None
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0) and self.table.item(r, 0).text() == tid:
                    existing = r
                    break
            if existing is not None:
                self.table.setItem(existing, 1, QtWidgets.QTableWidgetItem(t["cls"]))
                self.table.setItem(existing, 2, QtWidgets.QTableWidgetItem(f"{t['conf']:.2f}"))
                self.table.setItem(existing, 3, QtWidgets.QTableWidgetItem(t["fecha"]))
                self.table.setItem(existing, 4, QtWidgets.QTableWidgetItem(t["hora"]))
            else:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(tid))
                self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(t["cls"]))
                self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{t['conf']:.2f}"))
                self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(t["fecha"]))
                self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(t["hora"]))

        # Guarda tracks recientes (para CSV)
        if tracks:
            for t in tracks:
                self.report_data = [r for r in self.report_data if r["id"] != t["id"]]
                self.report_data.append(t)

        now = time.time()
        fps = 1.0 / (now - self.last_time) if now > self.last_time else 0.0
        self.last_time = now
        self.info_label.setText(
            f"FPS: {fps:.1f} | Tracks activos: {len(tracks)} | "
            f"Total detectados: {self.table.rowCount()} | "
            f"Último reporte: {self.last_report_name}"
        )

    def save_report(self):
        if not self.report_data:
            return
        uniq = {t["id"]: t for t in self.report_data}
        now = datetime.now()
        name = now.strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join(REPORT_DIR, f"tracks_{name}.csv")
        # 🔹 modo append si el archivo existe (histórico acumulativo)
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Clase", "Confianza", "Fecha", "Hora"])
            for t in uniq.values():
                writer.writerow([t["id"], t["cls"], f"{t['conf']:.2f}", t["fecha"], t["hora"]])
        self.report_data.clear()
        self.last_report_name = name
        print(f"💾 CSV actualizado: {filename}")

    def closeEvent(self, e):
        self.timer.stop()
        self.csv_timer.stop()
        self.worker.running = False
        self.thread.quit(); self.thread.wait()
        if self.cap:
            self.cap.release()
        e.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = YoloTrackingDashboard()
    win.show()
    sys.exit(app.exec())
