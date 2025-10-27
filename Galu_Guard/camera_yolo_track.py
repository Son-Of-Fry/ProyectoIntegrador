# camera_yolo_track.py — Qt + Ultralytics TRACK + colas + CSV + trayectorias
import sys, os, cv2, csv, json, time, torch, queue, threading
from math import hypot
from collections import defaultdict, deque
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO
import numpy as np

# ===================== CONFIGURACIÓN =====================
IS_WIN = sys.platform.startswith("win")
BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY

with open("config.json") as f:
    CONFIG = json.load(f)

# Parámetros YOLO / app
SELECTED_CLASSES = set(CONFIG["objects"])
CONF = CONFIG["yolo"]["confidence"]
IMGSZ = CONFIG["yolo"]["imgsz"]
EVERY = CONFIG["yolo"]["process_every"]
DEVICE = CONFIG["yolo"]["device"]
MODEL_PATH = CONFIG["yolo"]["model"]

# Tracker (puedes cambiar a "botsort.yaml" si prefieres)
TRACKER_YAML = "bytetrack.yaml"

# Política de guardado (en segundos entre fotos del mismo track)
SAVE_INTERVAL_SECS = 8

# Paths
os.makedirs("logs/images", exist_ok=True)
CSV_PATH = "logs/detections.csv"
if not os.path.isfile(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["timestamp", "cam_id", "class", "track_id", "conf", "x1", "y1", "x2", "y2", "image_path"]
        )

# ===================== OVERLAY DE ALERTAS =====================
class AlertOverlay(QtWidgets.QLabel):
    """Etiqueta superpuesta para mostrar alertas breves en la ventana."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "background-color: rgba(255,0,0,180); color: white; "
            "font-size: 18px; font-weight: bold; padding: 10px; border-radius: 8px;"
        )
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setVisible(False)

    def resizeEvent(self, event):
        if self.parent():
            w = self.parent().width()
            self.setGeometry(int(w * 0.1), 10, int(w * 0.8), 42)

    def show_alert(self, text, duration=1800):
        self.setText(text)
        self.setVisible(True)
        QtCore.QTimer.singleShot(duration, lambda: self.setVisible(False))

# ===================== WORKER DE TRACKING (QThread) =====================
class TrackingWorker(QtCore.QObject):
    """
    Hilo de tracking que consume frames desde una cola y emite resultados.
    Mantiene estado del tracker (persist=True) entre llamadas.
    """
    results_ready = QtCore.Signal(object, float)  # results, tiempo_ms

    def __init__(self, model, conf, imgsz, tracker_yaml, device, in_queue):
        super().__init__()
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.tracker_yaml = tracker_yaml
        self.device = device
        self.in_queue = in_queue
        self.running = True

    @QtCore.Slot()
    def run(self):
        """Bucle principal del hilo: toma frames de la cola, corre track, emite resultados."""
        # Ultralytics mantiene el estado de tracking con persist=True
        while self.running:
            item = self.in_queue.get()  # (frame, frame_ts)
            if item is None:
                break
            frame, frame_ts = item
            try:
                t0 = time.time()
                # Ejecuta tracking por frame; persist mantiene IDs entre invocaciones
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    tracker=self.tracker_yaml,
                    verbose=False
                )
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t_ms = (time.time() - t0) * 1000.0
                # results es una lista; usamos el primer elemento
                self.results_ready.emit(results[0], t_ms)
            except Exception as e:
                print("Error en tracking:", e)

# ===================== APLICACIÓN PRINCIPAL =====================
class CameraYoloTrackApp(QtWidgets.QWidget):
    """
    Ventana Qt: captura cámara, manda frames a una cola, recibe resultados del hilo de tracking,
    dibuja boxes e IDs persistentes, guarda CSV e imágenes según política, muestra trayectorias.
    """
    def __init__(self, cam_id):
        super().__init__()
        self.cam_id = cam_id
        self.setWindowTitle(f"Galu Guard (Track) - Cam {cam_id}")
        self.resize(900, 650)

        # Widget de video + overlay
        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.alert_overlay = AlertOverlay(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video)

        # Cámara
        self.cap = cv2.VideoCapture(cam_id, BACKEND)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Modelo YOLO
        self.model = YOLO(MODEL_PATH).to(DEVICE)
        if DEVICE == "cuda":
            try:
                self.model.fuse(); self.model.half()
            except Exception:
                pass

        # === Colas y worker de tracking ===
        self.in_queue = queue.Queue(maxsize=2)     # frames -> tracking
        self.worker = TrackingWorker(self.model, CONF, IMGSZ, TRACKER_YAML, DEVICE, self.in_queue)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.results_ready.connect(self.handle_results)
        self.thread.start()

        # Hilo para guardado de imágenes (no bloquear UI)
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # Estados para tracking
        self.frame_idx = 0
        self.last_frame = None
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # tid -> deque[(cx, cy)]
        self.track_meta = {}   # tid -> {"class": str, "first_saved": bool, "last_save": float}
        self.seen_once = set() # para alertas "nuevo ID"
        self.last_seen = {}    # tid -> last_ts (para futura lógica de desaparición)

        # Timer de captura (30-33ms ~ 30FPS)
        self.timer = QtCore.QTimer(self, interval=30)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    # ----------------- Bucle de captura -----------------
    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        self.frame_idx += 1
        self.last_frame = frame  # lo usamos para dibujar aunque el tracking procese cada N

        # Enviar a tracking cada N frames (para bajar carga)
        if self.frame_idx % EVERY == 0:
            # No bloquear si la cola está llena: descarta el más viejo (backpressure simple)
            try:
                if self.in_queue.full():
                    _ = self.in_queue.get_nowait()
                self.in_queue.put_nowait((frame.copy(), time.time()))
            except queue.Full:
                pass

        # Mostrar frame crudo (sin anotaciones) mientras llegan resultados
        self._show_qimage(frame)

    # ----------------- Manejar resultados del tracker -----------------
    def handle_results(self, result, t_ms):
        """
        result: ultralytics.engine.results.Results
        Contiene boxes.xyxy, boxes.id, boxes.conf, boxes.cls
        """
        if self.last_frame is None:
            return

        overlay = self.last_frame.copy()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        ids = result.boxes.id
        clss = result.boxes.cls
        confs = result.boxes.conf
        xyxy = result.boxes.xyxy

        if ids is None or len(ids) == 0:
            # Nada que dibujar, solo rendimiento
            self._draw_perf(overlay, t_ms)
            self._show_qimage(overlay)
            return

        ids = ids.int().cpu().tolist()
        clss = clss.int().cpu().tolist()
        confs = [float(c) for c in confs.cpu().tolist()]
        xyxy = xyxy.int().cpu().tolist()

        now = time.time()
        seen_now = set()

        for (tid, cls_i, conf, box) in zip(ids, clss, confs, xyxy):
            name = self.model.names[int(cls_i)]
            if name not in SELECTED_CLASSES:
                continue

            x1, y1, x2, y2 = box
            # Dibuja bbox + etiqueta con track_id
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                overlay, f"{name} TID:{tid} {conf:.2f}",
                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2
            )

            # Trayectoria (centro)
            cx, cy = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            self.track_history[tid].append((cx, cy))
            # Dibujar trayectoria del track_id (línea roja)
            if len(self.track_history[tid]) >= 2:
                pts_arr = np.array(self.track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts_arr], isClosed=False, color=(230, 0, 0), thickness=2)


            # Alerta por nuevo track_id
            if tid not in self.seen_once:
                self.seen_once.add(tid)
                self.alert_overlay.show_alert(f"Nuevo {name} (TID {tid})")

            # Registro/estado para guardado
            meta = self.track_meta.get(tid, {"class": name, "first_saved": False, "last_save": 0.0})
            # Guardar primera vez o por intervalo
            if (not meta["first_saved"]) or (now - meta["last_save"] >= SAVE_INTERVAL_SECS):
                # Crop y encolar guardado
                crop = overlay[y1:y2, x1:x2].copy()
                img_name = f"logs/images/cam{self.cam_id}_{timestamp}_{name}_tid{tid}.jpg"
                self.save_queue.put((crop, img_name))
                # CSV
                with open(CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [timestamp, self.cam_id, name, tid, round(conf, 2), x1, y1, x2, y2, img_name]
                    )
                # Actualiza meta
                meta["first_saved"] = True
                meta["last_save"] = now
                self.track_meta[tid] = meta

            # Marca visto ahora
            seen_now.add(tid)
            self.last_seen[tid] = now

        # (Opcional) Aquí podrías detectar desapariciones comparando track_meta keys vs seen_now

        # FPS/perf
        self._draw_perf(overlay, t_ms)
        self._show_qimage(overlay)

    # ----------------- Guardado asíncrono -----------------
    def _save_worker(self):
        while True:
            try:
                crop, img_name = self.save_queue.get()
                cv2.imwrite(img_name, crop)
                self.save_queue.task_done()
            except Exception as e:
                print("Error al guardar imagen:", e)

    # ----------------- Utilidades UI -----------------
    def _draw_perf(self, frame, t_ms):
        cv2.putText(frame, f"{t_ms:.1f} ms", (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    def _show_qimage(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

    # ----------------- Cierre limpio -----------------
    def closeEvent(self, e):
        try:
            self.timer.stop()
            # Parar hilo de tracking
            self.worker.running = False
            self.in_queue.put(None)  # centinela para salir del bucle
            self.thread.quit()
            self.thread.wait()
            # Esperar a que se guarden las imágenes en cola
            self.save_queue.join()
        except Exception as ex:
            print("Al cerrar:", ex)
        finally:
            if self.cap:
                self.cap.release()
        super().closeEvent(e)

# ===================== MAIN =====================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    windows = []
    for cam in CONFIG["cameras"]:
        w = CameraYoloTrackApp(cam["id"])
        w.show()
        windows.append(w)
        break  # Por ahora UNA cámara (como pediste). Quita este break si quieres abrir todas.
    app.exec()
