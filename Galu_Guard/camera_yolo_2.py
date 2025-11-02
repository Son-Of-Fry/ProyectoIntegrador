import sys, cv2, time, json, torch, os, csv, queue, threading
import numpy as np
import pyrealsense2 as rs
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO
from math import hypot
from collections import defaultdict

# ===================== CONFIGURACIÓN =====================
# Carga configuración desde el archivo config.json
IS_WIN = sys.platform.startswith("win")
BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY  # Selecciona backend según SO

with open("config.json") as f:
    CONFIG = json.load(f)

# Parámetros de detección cargados desde la configuración
SELECTED_CLASSES = set(CONFIG["objects"])
CONF = CONFIG["yolo"]["confidence"]
IMGSZ = CONFIG["yolo"]["imgsz"]
EVERY = CONFIG["yolo"]["process_every"]
DEVICE = CONFIG["yolo"]["device"]
MODEL_PATH = CONFIG["yolo"]["model"]

# Rutas de logs
os.makedirs("logs/images", exist_ok=True)
CSV_PATH = "logs/detections.csv"
RELATIONS_PATH = "logs/relations.csv"  # generado por el reportador (marcar “misma persona”)

# Si no existe el CSV, crea cabecera (10 columnas; no tocamos el esquema)
if not os.path.isfile(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "cam_id", "class", "id", "conf", "x1", "y1", "x2", "y2", "image_path"])


# ===================== UTILIDAD MEMORIA / IDENTIDADES =====================
def compute_hsv_hist(img_bgr):
    """
    Calcula un histograma HSV 8x8x8 normalizado (vector de 512 dims).
    Es rápido y suficiente para una ‘memoria’ ligera.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def hist_similarity(h1, h2):
    """Similitud por correlación (1.0 = idéntico)."""
    if h1 is None or h2 is None:
        return -1.0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


class IdentityMemory:
    """
    Carga relaciones (img1,img2,same) desde logs/relations.csv, crea clusters (‘Global IDs’)
    y mantiene histogramas de ejemplo por cluster para reconocer nuevas capturas.
    """
    def __init__(self, relations_path=RELATIONS_PATH, sim_threshold=0.9):
        self.relations_path = relations_path
        self.sim_threshold = sim_threshold
        self.cluster_id_by_img = {}            # path -> gid
        self.exemplars = defaultdict(list)     # gid -> [hist, ...]
        self._load_relations()

    def _load_relations(self):
        if not os.path.exists(self.relations_path):
            return

        # Union-Find simple para agrupar imágenes relacionadas
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Lee relaciones y une imágenes
        with open(self.relations_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 3: 
                    continue
                img1, img2, rel = row
                if rel.strip().lower() != "same":
                    continue
                union(img1, img2)

        # Mapea cada raíz a un gid incremental
        root_to_gid = {}
        next_gid = 1
        for img in list(parent.keys()):
            r = find(img)
            if r not in root_to_gid:
                root_to_gid[r] = next_gid
                next_gid += 1
            gid = root_to_gid[r]
            self.cluster_id_by_img[img] = gid

        # Construye histogramas de ejemplo por gid
        for img, gid in self.cluster_id_by_img.items():
            if os.path.exists(img):
                crop = cv2.imread(img)
                h = compute_hsv_hist(crop)
                if h is not None:
                    self.exemplars[gid].append(h)

    def match_gid(self, crop_bgr):
        """
        Devuelve el gid más parecido si alguna similitud supera el umbral;
        de lo contrario, retorna 0 (desconocido).
        """
        if not self.exemplars:
            return 0
        h = compute_hsv_hist(crop_bgr)
        if h is None:
            return 0

        best_gid, best_sim = 0, -1.0
        for gid, hists in self.exemplars.items():
            # tomamos el máximo contra los ejemplares del cluster
            for eh in hists:
                s = hist_similarity(h, eh)
                if s > best_sim:
                    best_sim, best_gid = s, gid
        return best_gid if best_sim >= self.sim_threshold else 0

    def note_new_example(self, gid, crop_bgr):
        """(opcional) añadir como ejemplo en memoria un recorte reconocido."""
        if gid <= 0:
            return
        h = compute_hsv_hist(crop_bgr)
        if h is not None:
            self.exemplars[gid].append(h)


# ===================== CENTROID TRACKER =====================
class CentroidTracker:
    """Asigna IDs persistentes a objetos detectados según cercanía de centroides."""

    def __init__(self, dist_thresh=60, max_age=10):
        self.next_id = 1               # ID siguiente a asignar
        self.objects = {}              # Diccionario: id -> (cx, cy, age)
        self.dist_thresh = dist_thresh # Distancia máxima para considerar mismo objeto
        self.max_age = max_age         # Cuántos frames mantener un objeto sin actualizar

    def update(self, detections):
        new_objects = {}
        centroids = [(int((x1+x2)//2), int((y1+y2)//2)) for (x1, y1, x2, y2) in detections]
        used_ids = set()

        # Intenta asociar nuevos centroides con objetos previos
        for cid, (cx_old, cy_old, age) in list(self.objects.items()):
            min_dist, best_idx = 1e9, -1
            for i, (cx, cy) in enumerate(centroids):
                d = hypot(cx - cx_old, cy - cy_old)
                if d < min_dist:
                    min_dist, best_idx = d, i
            # Si la distancia es menor al umbral, se considera el mismo objeto
            if best_idx != -1 and min_dist < self.dist_thresh:
                new_objects[cid] = (*centroids[best_idx], 0)
                used_ids.add(best_idx)
            else:
                # Incrementa edad; elimina si supera max_age
                if age + 1 < self.max_age:
                    new_objects[cid] = (cx_old, cy_old, age + 1)

        # Crea nuevos IDs para objetos no asociados
        for i, (cx, cy) in enumerate(centroids):
            if i not in used_ids:
                new_objects[self.next_id] = (cx, cy, 0)
                self.next_id += 1

        self.objects = new_objects
        return self.objects


# ===================== HILO DE INFERENCIA =====================
class InferenceWorker(QtCore.QObject):
    """Hilo que ejecuta inferencias YOLO sin bloquear la interfaz Qt."""

    result_ready = QtCore.Signal(object, float)  # Señal: resultados + tiempo ms
    run_request = QtCore.Signal(object)          # Señal: solicita nueva inferencia

    def __init__(self, model, conf, imgsz, device):
        super().__init__()
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.running = True
        self.run_request.connect(self._do_inference)

    @QtCore.Slot(object)
    def _do_inference(self, frame):
        """Ejecuta inferencia y emite resultados."""
        if not self.running:
            return
        t0 = time.time()
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        t_ms = (time.time() - t0) * 1000
        self.result_ready.emit(results, t_ms)


# ===================== SISTEMA DE ALERTAS =====================
class AlertOverlay(QtWidgets.QLabel):
    """Etiqueta superpuesta que muestra mensajes de alerta sobre el video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Estilo visual: fondo rojo semitransparente y texto blanco centrado
        self.setStyleSheet("background-color: rgba(255, 0, 0, 180); color: white; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 8px;")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setVisible(False)

    def resizeEvent(self, event):
        """Reajusta tamaño y posición del overlay cuando cambia el tamaño de ventana."""
        if self.parent():
            w = self.parent().width()
            self.setGeometry(int(w*0.1), 10, int(w*0.8), 50)

    def show_alert(self, text, duration=2000):
        """Muestra un mensaje temporal de alerta durante 'duration' ms."""
        self.setText(text)
        self.setVisible(True)
        QtCore.QTimer.singleShot(duration, lambda: self.setVisible(False))


# ===================== APLICACIÓN PRINCIPAL =====================
class CameraYoloApp(QtWidgets.QWidget):
    """Ventana principal: captura cámara, ejecuta YOLO y muestra resultados."""

    def __init__(self, cam_id):
        super().__init__()
        self.cam_id = cam_id
        self.setWindowTitle(f"Galu Guard - Cámara {cam_id}")
        self.resize(800, 600)

        # Elementos visuales principales
        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.alert_overlay = AlertOverlay(self)
        self.alert_overlay.resize(self.width(), 50)
        self.alert_overlay.move(int(self.width()*0.1), 10)

        # Layout principal (solo contiene el video)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video)

        # Configuración de la cámara según tipo (USB, RTSP, REALSENSE)
        cam_config = next((c for c in CONFIG.get("cameras", []) if str(c.get("id")) == str(cam_id)), None)
        self.cam_type = None
        if cam_config:
            self.cam_type = cam_config.get("type", "USB").upper()
            if self.cam_type == "RTSP":
                source = cam_config.get("url")
                print(f"🎥 Iniciando cámara RTSP: {source}")
                self.cap = cv2.VideoCapture(source, BACKEND)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if not self.cap.isOpened():
                    print(f"⚠️ No se pudo abrir RTSP {source}, reintentando con backend genérico...")
                    self.cap = cv2.VideoCapture(source)
            elif self.cam_type == "REAL SENSE" or self.cam_type == "REALSENSE":
                print("🎥 Iniciando cámara Intel RealSense usando pyrealsense2")
                self.pipeline = rs.pipeline()
                self.rs_config = rs.config()
                self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(self.rs_config)
            else:  # USB por defecto
                print(f"🎥 Iniciando cámara USB índice {cam_id}")
                self.cap = cv2.VideoCapture(int(cam_id), BACKEND)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            print(f"⚠️ No se encontró configuración para la cámara {cam_id}, usando índice directo.")
            self.cam_type = "USB"
            self.cap = cv2.VideoCapture(int(cam_id), BACKEND)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Carga modelo YOLO
        self.model = YOLO(MODEL_PATH).to(DEVICE)
        if DEVICE == "cuda":
            try:
                self.model.fuse(); self.model.half()  # Optimización si hay GPU
            except: 
                pass

        # Inicializa tracker y memoria de identidad
        self.tracker = CentroidTracker(dist_thresh=80, max_age=15)
        self.identity_memory = IdentityMemory(RELATIONS_PATH, sim_threshold=0.9)

        # Para no spamear alertas de la misma identidad global
        self.seen_global_ids = set()
        self.recent_alerts = set()

        # Configura hilo de inferencia
        self.worker = InferenceWorker(self.model, CONF, IMGSZ, DEVICE)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.start()
        self.worker.result_ready.connect(self.handle_results)

        # Crea buffer asincrónico de guardado de imágenes
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # Inicia bucle de actualización de frames
        self.frame_idx = 0
        self.timer = QtCore.QTimer(self, interval=30)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    # ===================== ACTUALIZAR FRAME =====================
    def update_frame(self):
        """Lee la cámara y actualiza la imagen mostrada; envía frames a inferencia periódicamente."""
        frame = None
        ok = False
        if hasattr(self, 'cam_type') and (self.cam_type == "REAL SENSE" or self.cam_type == "REALSENSE"):
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    ok = True
            except Exception as e:
                print("Error al obtener frame de RealSense:", e)
                ok = False
        else:
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        self.frame_idx += 1
        self.current_frame = frame.copy()

        # Cada N frames lanza inferencia en segundo plano
        if self.frame_idx % EVERY == 0:
            self.worker.run_request.emit(frame)

        # Convierte frame a formato Qt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    # ===================== RESULTADOS =====================
    def handle_results(self, results, t_ms):
        """Dibuja detecciones, gestiona memoria y alertas."""
        overlay = self.current_frame.copy()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            name = self.model.names[int(cls)]
            if name not in SELECTED_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append((name, float(conf), (x1, y1, x2, y2)))

        # Actualiza seguimiento de objetos
        tracked = self.tracker.update([b for (_, _, b) in detections])

        # Empata detecciones con IDs del tracker (por orden)
        for (name, conf, (x1, y1, x2, y2)), (tid, (cx, cy, age)) in zip(detections, list(tracked.items())):
            # Dibuja
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ID:{tid}"
            cv2.putText(overlay, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Recorte para memoria/guardado
            crop = self.current_frame[y1:y2, x1:x2]

            # ====== MEMORIA: intenta reconocer GID por similitud con relations.csv ======
            gid = self.identity_memory.match_gid(crop)  # 0 = desconocido
            if gid > 0:
                # si reconocido y no alertado recientemente, muestra alerta “Reconocido”
                if gid not in self.seen_global_ids:
                    self.seen_global_ids.add(gid)
                    self.alert_overlay.show_alert(f"Reconocido {name} (GID {gid})")
                # opcional: enriquecer memoria con el nuevo recorte
                self.identity_memory.note_new_example(gid, crop)
            else:
                # solo alertamos ‘Nuevo’ si NO fue visto antes (por tracker local)
                if tid not in self.recent_alerts:
                    self.recent_alerts.add(tid)
                    self.alert_overlay.show_alert(f"Nuevo {name} detectado (ID {tid})")

            # Nombre del archivo incluye gid para futura referencia (sin cambiar CSV)
            gid_tag = f"_gid{gid}" if gid > 0 else "_gid0"
            img_name = f"logs/images/cam{self.cam_id}_{timestamp}_{name}_id{tid}{gid_tag}.jpg"

            # Guardado asíncrono
            self.save_queue.put((crop, img_name))

            # Registra en CSV (mantenemos 10 columnas)
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, self.cam_id, name, tid, round(conf, 2), x1, y1, x2, y2, img_name])

        # Muestra tiempo de inferencia
        cv2.putText(overlay, f"{t_ms:.1f} ms", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Convierte frame final a formato Qt
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    # ===================== BUFFER DE GUARDADO =====================
    def _save_worker(self):
        """Hilo dedicado a guardar imágenes sin bloquear el hilo principal."""
        while True:
            try:
                crop, img_name = self.save_queue.get()
                cv2.imwrite(img_name, crop)
                self.save_queue.task_done()
            except Exception as e:
                print("Error al guardar imagen:", e)

    # ===================== CIERRE =====================
    def closeEvent(self, e):
        """Limpieza completa al cerrar la ventana."""
        self.timer.stop()
        self.worker.running = False
        self.thread.quit(); self.thread.wait()
        self.save_queue.join()
        if self.cap:
            self.cap.release()
        import deduplicator
        deduplicator.deduplicate()  # Ejecuta limpieza final de CSV
        super().closeEvent(e)


# ===================== MAIN =====================
if __name__ == "__main__":
    # Crea aplicación Qt y lanza una ventana por cámara configurada
    app = QtWidgets.QApplication(sys.argv)
    windows = []
    for cam in CONFIG.get("cameras", []):
        source_id = cam.get("id", 0)
        print(f"🎥 Cargando cámara '{cam.get('name', 'Sin nombre')}' tipo {cam.get('type', 'USB')} id={source_id}")
        w = CameraYoloApp(source_id)
        w.show()
        windows.append(w)
    if not CONFIG.get("cameras"):
        print("⚠️ No hay cámaras definidas en config.json")
    app.exec()
