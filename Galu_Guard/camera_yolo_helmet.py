# camera_yolo_helmet.py — Detección especializada de cascos (modelo fine-tuneado)
import sys, cv2, time, json, torch, os, csv
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

IS_WIN = sys.platform.startswith("win")
BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY

# Leer configuración desde config.json (modelo fine-tuneado debe estar ahí)
with open("config.json") as f:
    CONFIG = json.load(f)

SELECTED_CLASSES = set(CONFIG["objects"])
CONF = CONFIG["yolo"]["confidence"]
IMGSZ = CONFIG["yolo"]["imgsz"]
EVERY = CONFIG["yolo"]["process_every"]
DEVICE = CONFIG["yolo"]["device"]
MODEL_PATH = CONFIG["yolo"]["model"]

os.makedirs("logs/images", exist_ok=True)
CSV_PATH = "logs/detections.csv"
if not os.path.isfile(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "cam_id", "class", "conf", "x1", "y1", "x2", "y2", "image_path"])

class CameraYoloApp(QtWidgets.QWidget):
    def __init__(self, cam_id):
        super().__init__()
        self.cam_id = cam_id
        self.setWindowTitle("Galu Guard — Detección de Cascos")
        self.resize(800, 600)
        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video)

        self.cap = cv2.VideoCapture(cam_id, BACKEND)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.model = YOLO(MODEL_PATH).to(DEVICE)
        if DEVICE == "cuda":
            try: self.model.fuse(); self.model.half()
            except: pass

        self.frame_idx = 0
        self.timer = QtCore.QTimer(self, interval=30)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok: return
        self.frame_idx += 1
        overlay = frame.copy()

        if self.frame_idx % EVERY == 0:
            t0 = time.time()
            results = self.model(overlay, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            if DEVICE == "cuda": torch.cuda.synchronize()
            t_ms = (time.time() - t0) * 1000

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                name = self.model.names[int(cls)]
                if name not in SELECTED_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, f"{name} {conf:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Guardar recorte
                crop = frame[y1:y2, x1:x2]
                img_name = f"logs/images/cam{self.cam_id}_{timestamp}_{name}.jpg"
                cv2.imwrite(img_name, crop)

                # Guardar en CSV
                with open(CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([timestamp, self.cam_id, name, round(float(conf), 2),
                                            x1, y1, x2, y2, img_name])

            cv2.putText(overlay, f"{t_ms:.1f} ms", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def closeEvent(self, e):
        self.timer.stop()
        if self.cap: self.cap.release()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    windows = []
    for cam in CONFIG["cameras"]:
        w = CameraYoloApp(cam["id"])
        w.show()
        windows.append(w)
    app.exec()

    # Desduplicación al cerrar
    import deduplicator
    deduplicator.deduplicate()
