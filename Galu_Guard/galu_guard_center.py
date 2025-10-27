# galu_guard_center_v2.py — Galu Guard Control Center v2
# - Cámaras con combo para elegir fuente y botón "Iniciar análisis"
# - Preconfiguración por cámara antes de iniciar (diálogo)
# - Dashboard con exportación PDF+ZIP y auto-refresco
# - Galería con filtros, eliminar y "misma persona"
# - Memoria con grupos (GID) y miniaturas; limpiar memoria
# - Configuración global y por cámara editable (form) + recargar

import sys, os, csv, json, time, glob, zipfile
from datetime import datetime
from collections import defaultdict, deque

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

# Reporte PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ===================== RUTAS BÁSICAS =====================
CONFIG_PATH = "config.json"
IMAGES_DIR = "logs/images"
CSV_PATH = "logs/detections.csv"         # tracking => sin deduplicador
REL_PATH = "logs/relations.csv"
REPORTS_DIR = "logs/reportes"
os.makedirs("logs", exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# CSV base
if not os.path.isfile(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp","cam_id","class","track_id","conf","x1","y1","x2","y2","image_path"])

# ===================== UTILIDADES =====================
def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def load_csv_rows(path=CSV_PATH):
    if not os.path.exists(path): return []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[1:] if len(rows) > 1 else []

def export_pdf(pdf_path, rows):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    elements.append(Paragraph("<b>Reporte de Detecciones — Galu Guard</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # Resumen por clase / cámara
    summary = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if len(r) < 10: continue
        _, cam, cls, *_ = r
        summary[cls][cam] += 1
    data = [["Clase","Cámara","Conteo"]]
    total = 0
    for cls, cams in summary.items():
        for cam, count in cams.items():
            data.append([cls, cam, str(count)])
            total += count
    data.append(["", "Total", str(total)])
    table = Table(data, colWidths=[180, 120, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a82da")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 16))

    # Últimos eventos
    elements.append(Paragraph("<b>Últimas 20 detecciones</b>", styles["Heading2"]))
    for r in rows[-20:]:
        try:
            ts, cam, cls, tid, conf, *_ = r
            elements.append(Paragraph(f"[{ts}] Cam{cam} → {cls} (TID {tid}, {conf})", styles["Code"]))
        except:
            pass
    doc.build(elements)

def make_zip(base_name, pdf_path, csv_path, images_dir):
    zip_path = f"{base_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(pdf_path): z.write(pdf_path, os.path.basename(pdf_path))
        if os.path.exists(csv_path): z.write(csv_path, os.path.basename(csv_path))
        for root, _, files in os.walk(images_dir):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, os.path.dirname(images_dir))
                z.write(p, rel)
    return zip_path

# =============== MEMORIA (relations.csv → grupos GID) ===============
def save_relations(pairs):
    os.makedirs(os.path.dirname(REL_PATH), exist_ok=True)
    with open(REL_PATH, "a", newline="") as f:
        w = csv.writer(f)
        for a,b in pairs:
            w.writerow([a,b,"same"])

def load_relations():
    if not os.path.exists(REL_PATH): return []
    with open(REL_PATH, newline="") as f:
        return [r for r in csv.reader(f) if len(r)>=3 and r[2]=="same"]

def build_gid_groups():
    # Union-Find básico
    rels = load_relations()
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a,b):
        pa, pb = find(a), find(b)
        if pa != pb: parent[pb] = pa
    for a,b,_ in rels:
        union(a,b)
    gid_by_root = {}
    gid_map = {}
    next_gid = 1
    for k in list(parent.keys()):
        root = find(k)
        if root not in gid_by_root:
            gid_by_root[root] = f"G{next_gid}"; next_gid += 1
        gid_map[k] = gid_by_root[root]
    # gid -> lista de imgs
    groups = defaultdict(list)
    for path, gid in gid_map.items():
        groups[gid].append(path)
    return groups   # {G1: [img1, img2,...], ...}

# ===================== UI: Overlay de alertas =====================
class AlertOverlay(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(255,0,0,180); color:#fff; font-weight:700; padding:8px; border-radius:8px;")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setVisible(False)
    def resizeEvent(self, e):
        if self.parent():
            w = self.parent().width()
            self.setGeometry(int(w*0.1), 10, int(w*0.8), 36)
    def show_alert(self, text, duration=1500):
        self.setText(text); self.setVisible(True)
        QtCore.QTimer.singleShot(duration, lambda: self.setVisible(False))

# ===================== Worker de Tracking =====================
class TrackingWorker(QtCore.QObject):
    results_ready = QtCore.Signal(object, float)
    def __init__(self, model, conf, imgsz, tracker_yaml, device, in_queue):
        super().__init__()
        self.model = model; self.conf = conf; self.imgsz = imgsz
        self.tracker_yaml = tracker_yaml; self.device = device
        self.in_queue = in_queue; self.running = True
    @QtCore.Slot()
    def run(self):
        while self.running:
            item = self.in_queue.get()
            if item is None: break
            frame, _ts = item
            try:
                t0 = time.time()
                results = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz,
                                           tracker=self.tracker_yaml, verbose=False)
                if self.device == "cuda":
                    import torch; torch.cuda.synchronize()
                t_ms = (time.time() - t0) * 1000.0
                self.results_ready.emit(results[0], t_ms)
            except Exception as e:
                print("Tracking error:", e)

# ===================== Widget de Cámara =====================
class CameraYoloWidget(QtWidgets.QFrame):
    def __init__(self, cam_source, config):
        super().__init__()
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Raised)
        self.cam_source = cam_source
        self.cfg = config
        self.SELECTED_CLASSES = set(config["objects"])
        y = config["yolo"]
        self.CONF = y["confidence"]; self.IMGSZ = y["imgsz"]; self.EVERY = y["process_every"]
        self.DEVICE = y["device"]; self.MODEL_PATH = y["model"]
        self.TRACKER_YAML = "bytetrack.yaml"
        self.SAVE_INTERVAL_SECS = 8

        v = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel(f"Cámara {cam_source}")
        self.title.setStyleSheet("font-weight:700")
        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(420, 240)
        self.alert_overlay = AlertOverlay(self)
        v.addWidget(self.title); v.addWidget(self.video)

        # Cámara
        IS_WIN = sys.platform.startswith("win")
        BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam_source, BACKEND)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Modelo
        self.model = YOLO(self.MODEL_PATH).to(self.DEVICE)
        if self.DEVICE == "cuda":
            try: self.model.fuse(); self.model.half()
            except: pass

        import queue as pyqueue
        self.in_pyqueue = pyqueue.Queue(maxsize=2)
        self.worker = TrackingWorker(self.model, self.CONF, self.IMGSZ, self.TRACKER_YAML,
                                     self.DEVICE, self.in_pyqueue)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.results_ready.connect(self.handle_results)
        self.thread.start()

        # Guardado async
        self.save_queue = pyqueue.Queue()
        import threading
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # Estado
        self.frame_idx = 0
        self.last_frame = None
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_meta = {}  # tid -> {"first_saved":bool,"last_save":float,"class":str}
        self.seen_once = set()

        # Timer captura
        self.timer = QtCore.QTimer(self, interval=30)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok: return
        self.frame_idx += 1
        self.last_frame = frame
        if self.frame_idx % self.EVERY == 0:
            try:
                if self.in_pyqueue.full(): _ = self.in_pyqueue.get_nowait()
                self.in_pyqueue.put_nowait((frame.copy(), time.time()))
            except: pass
        self._show_qimage(frame)

    def handle_results(self, result, t_ms):
        if self.last_frame is None: return
        overlay = self.last_frame.copy()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        ids = result.boxes.id
        if ids is None or len(ids)==0:
            self._draw_perf(overlay, t_ms); self._show_qimage(overlay); return

        ids = ids.int().cpu().tolist()
        clss = result.boxes.cls.int().cpu().tolist()
        confs = [float(c) for c in result.boxes.conf.cpu().tolist()]
        xyxy = result.boxes.xyxy.int().cpu().tolist()

        now = time.time()
        for (tid, cls_i, conf, box) in zip(ids, clss, confs, xyxy):
            name = self.model.names[int(cls_i)]
            if name not in self.SELECTED_CLASSES: continue
            x1,y1,x2,y2 = box
            # bbox
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,220,0), 2)
            cv2.putText(overlay, f"{name} TID:{tid} {conf:.2f}", (x1, max(20,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0,220,0), 2)
            # trayectoria
            cx,cy = int((x1+x2)/2), int((y1+y2)/2)
            self.track_history[tid].append((cx,cy))
            if len(self.track_history[tid])>=2:
                pts = np.array(self.track_history[tid], dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(overlay, [pts], False, (230,0,0), 2)

            # alerta primera vez
            if tid not in self.seen_once:
                self.seen_once.add(tid)
                self.alert_overlay.show_alert(f"Nuevo {name} (TID {tid})")

            # guardar (primera vez o cada N seg)
            meta = self.track_meta.get(tid, {"first_saved":False,"last_save":0.0,"class":name})
            if (not meta["first_saved"]) or (now - meta["last_save"] >= self.SAVE_INTERVAL_SECS):
                crop = overlay[y1:y2, x1:x2].copy()
                img_name = f"{IMAGES_DIR}/cam{self.cam_source}_{timestamp}_{name}_tid{tid}.jpg"
                self.save_queue.put((crop, img_name))
                with open(CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([timestamp, self.cam_source, name, tid, round(conf,2), x1,y1,x2,y2, img_name])
                meta["first_saved"] = True; meta["last_save"] = now
                self.track_meta[tid] = meta

        self._draw_perf(overlay, t_ms)
        self._show_qimage(overlay)

    def _save_worker(self):
        while True:
            try:
                crop, path = self.save_queue.get()
                cv2.imwrite(path, crop)
                self.save_queue.task_done()
            except Exception as e:
                print("Save error:", e)

    def _draw_perf(self, frame, t_ms):
        cv2.putText(frame, f"{t_ms:.1f} ms", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    def _show_qimage(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(self.video.size(),
                            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def close(self):
        try:
            self.timer.stop()
            self.worker.running = False
            self.in_pyqueue.put(None)
            self.thread.quit(); self.thread.wait()
        except: pass
        if self.cap: self.cap.release()
        super().close()

# ===================== Diálogo de configuración por cámara =====================
class CameraConfigDialog(QtWidgets.QDialog):
    """Permite editar parámetros antes de iniciar una cámara."""
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar cámara")
        self.cfg = json.loads(json.dumps(cfg))  # copia
        y = self.cfg["yolo"]

        form = QtWidgets.QFormLayout(self)
        self.model = QtWidgets.QLineEdit(y["model"])
        self.conf = QtWidgets.QDoubleSpinBox(); self.conf.setRange(0.0,1.0); self.conf.setSingleStep(0.01); self.conf.setValue(y["confidence"])
        self.imgsz = QtWidgets.QSpinBox(); self.imgsz.setRange(160, 1920); self.imgsz.setValue(y["imgsz"])
        self.pevery = QtWidgets.QSpinBox(); self.pevery.setRange(1, 30); self.pevery.setValue(y["process_every"])
        self.device = QtWidgets.QComboBox(); self.device.addItems(["cpu","cuda"]); 
        idx = self.device.findText(y["device"]); 
        self.device.setCurrentIndex(idx if idx>=0 else 0)

        form.addRow("Modelo (.pt):", self.model)
        form.addRow("Confianza:", self.conf)
        form.addRow("ImgSz:", self.imgsz)
        form.addRow("Process every:", self.pevery)
        form.addRow("Device:", self.device)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self):
        v = json.loads(json.dumps(self.cfg))
        v["yolo"]["model"] = self.model.text().strip()
        v["yolo"]["confidence"] = float(self.conf.value())
        v["yolo"]["imgsz"] = int(self.imgsz.value())
        v["yolo"]["process_every"] = int(self.pevery.value())
        v["yolo"]["device"] = self.device.currentText()
        return v

# ===================== Tabs: Dashboard / Galería / Memoria / Config =====================
class DashboardTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        self.summary_table = QtWidgets.QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Clase","Cámara","Conteo"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)

        self.log_box = QtWidgets.QTextEdit(readOnly=True)
        self.log_box.setStyleSheet("background-color:#111;color:#0f0;font-family:monospace;")

        btns = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("🔄 Actualizar")
        self.export_btn = QtWidgets.QPushButton("📦 Exportar (PDF+ZIP)")
        btns.addWidget(self.refresh_btn); btns.addWidget(self.export_btn)

        v.addWidget(QtWidgets.QLabel("<b>Resumen de detecciones</b>"))
        v.addWidget(self.summary_table)
        v.addWidget(QtWidgets.QLabel("<b>Últimos eventos</b>"))
        v.addWidget(self.log_box)
        v.addLayout(btns)

        self.refresh_btn.clicked.connect(self.refresh)
        self.export_btn.clicked.connect(self.export_report)

        # Auto-refresco
        self.timer = QtCore.QTimer(self, interval=5000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        rows = load_csv_rows()
        conteo = defaultdict(lambda: defaultdict(int))
        last = []
        for r in rows:
            try:
                ts, cam, cls, tid, conf, *_ = r
                conteo[cls][cam] += 1
                last.append(f"[{ts}] Cam{cam} → {cls} (TID {tid}, {conf})")
            except: pass
        self.summary_table.setRowCount(0)
        for cls, cams in conteo.items():
            for cam, count in cams.items():
                r = self.summary_table.rowCount()
                self.summary_table.insertRow(r)
                self.summary_table.setItem(r, 0, QtWidgets.QTableWidgetItem(cls))
                self.summary_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(cam)))
                self.summary_table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(count)))
        self.log_box.setText("\n".join(reversed(last[-50:])))

    def export_report(self):
        rows = load_csv_rows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "Sin datos", "No hay datos para exportar."); return
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        pdf_path = os.path.join(REPORTS_DIR, f"reporte_{ts}.pdf")
        export_pdf(pdf_path, rows)
        zip_base = os.path.join(REPORTS_DIR, f"reporte_{ts}")
        zip_path = make_zip(zip_base, pdf_path, CSV_PATH, IMAGES_DIR)
        QtWidgets.QMessageBox.information(self, "Exportado", f"PDF: {pdf_path}\nZIP: {zip_path}")

class GalleryTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        filters = QtWidgets.QHBoxLayout()
        self.class_filter = QtWidgets.QLineEdit(); self.class_filter.setPlaceholderText("Filtrar por clase (ej. person)")
        self.cam_filter = QtWidgets.QLineEdit(); self.cam_filter.setPlaceholderText("Filtrar por cámara (ej. 0)")
        self.apply_btn = QtWidgets.QPushButton("Aplicar filtro")
        self.apply_btn.clicked.connect(self.reload_gallery)
        filters.addWidget(self.class_filter); filters.addWidget(self.cam_filter); filters.addWidget(self.apply_btn)

        self.gallery = QtWidgets.QListWidget()
        self.gallery.setViewMode(QtWidgets.QListView.IconMode)
        self.gallery.setIconSize(QtCore.QSize(160,120))
        self.gallery.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.gallery.setSpacing(8)
        self.gallery.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.gallery.itemSelectionChanged.connect(self.show_detail)

        self.detail = QtWidgets.QTextEdit(readOnly=True)
        self.detail.setStyleSheet("background-color:#111;color:#0f0;font-family:monospace;")

        btns = QtWidgets.QHBoxLayout()
        self.delete_btn = QtWidgets.QPushButton("🗑️ Eliminar")
        self.same_btn = QtWidgets.QPushButton("🧠 Misma persona")
        self.refresh_btn = QtWidgets.QPushButton("🔄 Actualizar")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.same_btn.clicked.connect(self.mark_same_person)
        self.refresh_btn.clicked.connect(self.reload_gallery)
        btns.addWidget(self.delete_btn); btns.addWidget(self.same_btn); btns.addWidget(self.refresh_btn)

        v.addLayout(filters); v.addWidget(self.gallery); v.addWidget(self.detail); v.addLayout(btns)
        self.csv_index = {}
        self.reload_gallery()

    def reload_gallery(self):
        rows = load_csv_rows()
        self.csv_index = {}
        for r in rows:
            if len(r) < 10: continue
            ts, cam, cls, tid, conf, x1,y1,x2,y2, img = r
            self.csv_index[img] = (ts, cam, cls, tid, conf, x1,y1,x2,y2)

        files = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        files.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)

        f_cls = self.class_filter.text().strip().lower()
        f_cam = self.cam_filter.text().strip()
        self.gallery.clear()
        for p in files:
            meta = self.csv_index.get(p)
            cls,cam = (meta[2],meta[1]) if meta else (None,None)
            if f_cls and (not cls or f_cls not in str(cls).lower()): continue
            if f_cam and (str(cam) != f_cam): continue
            item = QtWidgets.QListWidgetItem()
            icon = QtGui.QIcon(p) if os.path.exists(p) else QtGui.QIcon.fromTheme("image-missing")
            item.setIcon(icon); item.setToolTip(p); item.setText(f"{cls or 'unknown'}\nCam{cam or '?'}")
            item.setData(QtCore.Qt.UserRole, meta)
            self.gallery.addItem(item)

    def show_detail(self):
        sel = self.gallery.selectedItems()
        if not sel:
            self.detail.setText("Selecciona imágenes para ver detalles"); return
        lines = []
        for it in sel:
            p = it.toolTip()
            meta = it.data(QtCore.Qt.UserRole)
            lines.append(f"Ruta: {p}")
            if meta:
                ts, cam, cls, tid, conf, *_ = meta
                lines.append(f"  ts:{ts} cam:{cam} cls:{cls} tid:{tid} conf:{conf}")
        self.detail.setText("\n".join(lines))

    def delete_selected(self):
        sel = self.gallery.selectedItems()
        if not sel: return
        ok = QtWidgets.QMessageBox.question(self, "Confirmar", f"¿Eliminar {len(sel)} imagen(es) y su registro?",
                                            QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
        if ok != QtWidgets.QMessageBox.Yes: return
        to_delete = {it.toolTip() for it in sel}
        # borrar archivos
        for p in to_delete:
            try:
                if os.path.exists(p): os.remove(p)
            except Exception as e:
                print("Delete error:", e)
        # reescribir CSV
        rows = load_csv_rows()
        header = ["timestamp","cam_id","class","track_id","conf","x1","y1","x2","y2","image_path"]
        keep = [r for r in rows if len(r)>=10 and r[9] not in to_delete]
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(keep)
        self.reload_gallery()
        QtWidgets.QMessageBox.information(self, "Listo", "Imágenes eliminadas.")

    def mark_same_person(self):
        sel = self.gallery.selectedItems()
        if len(sel) < 2:
            QtWidgets.QMessageBox.information(self, "Selecciona más", "Elige al menos 2 imágenes."); return
        paths = [it.toolTip() for it in sel]
        pairs = []
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                pairs.append((paths[i], paths[j]))
        save_relations(pairs)
        QtWidgets.QMessageBox.information(self, "Guardado", "Relación agregada (relations.csv).")

class MemoryTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["GID","Ejemplo","Miembros"])
        self.table.horizontalHeader().setStretchLastSection(True)

        h = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("🔄 Reconstruir memoria")
        self.clear_btn = QtWidgets.QPushButton("🧹 Limpiar memoria")
        h.addWidget(self.refresh_btn); h.addWidget(self.clear_btn)
        self.refresh_btn.clicked.connect(self.refresh)
        self.clear_btn.clicked.connect(self.clear_memory)

        v.addWidget(QtWidgets.QLabel("<b>Grupos de identidad (memoria)</b>"))
        v.addWidget(self.table); v.addLayout(h)
        self.refresh()

    def refresh(self):
        groups = build_gid_groups()   # {G1: [img1,img2,...]}
        self.table.setRowCount(0)
        for gid, imgs in groups.items():
            r = self.table.rowCount()
            self.table.insertRow(r)
            # GID
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(gid))
            # Ejemplo (miniatura)
            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignCenter)
            if imgs and os.path.exists(imgs[0]):
                pix = QtGui.QPixmap(imgs[0]).scaled(120, 90, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                label.setPixmap(pix)
            self.table.setCellWidget(r, 1, label)
            # Lista corta
            txt = "\n".join(os.path.basename(p) for p in imgs[:6])
            if len(imgs) > 6: txt += f"\n+{len(imgs)-6} más..."
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(txt))

    def clear_memory(self):
        if not os.path.exists(REL_PATH):
            QtWidgets.QMessageBox.information(self, "Memoria", "No hay memoria para limpiar."); return
        ok = QtWidgets.QMessageBox.question(self, "Confirmar", "¿Borrar relations.csv?",
                                            QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
        if ok != QtWidgets.QMessageBox.Yes: return
        try:
            os.remove(REL_PATH)
        except: pass
        self.refresh()
        QtWidgets.QMessageBox.information(self, "Memoria", "Memoria borrada.")

class ConfigTab(QtWidgets.QWidget):
    """Editor de config global y por cámara (form)."""
    def __init__(self, reload_cb):
        super().__init__()
        self.reload_cb = reload_cb
        v = QtWidgets.QVBoxLayout(self)

        # Vista JSON
        v.addWidget(QtWidgets.QLabel("<b>Configuración (JSON)</b>"))
        self.json_view = QtWidgets.QTextEdit(readOnly=True)
        self.json_view.setStyleSheet("background-color:#111;color:#0f0;font-family:monospace;")
        v.addWidget(self.json_view)

        # Form por cámara
        v.addWidget(QtWidgets.QLabel("<b>Editar por cámara</b>"))
        form = QtWidgets.QFormLayout()
        self.cam_combo = QtWidgets.QComboBox()
        self.model = QtWidgets.QLineEdit()
        self.conf = QtWidgets.QDoubleSpinBox(); self.conf.setRange(0,1); self.conf.setSingleStep(0.01)
        self.imgsz = QtWidgets.QSpinBox(); self.imgsz.setRange(160,1920)
        self.pevery = QtWidgets.QSpinBox(); self.pevery.setRange(1,30)
        self.device = QtWidgets.QComboBox(); self.device.addItems(["cpu","cuda"])
        form.addRow("Cámara:", self.cam_combo)
        form.addRow("Modelo (.pt):", self.model)
        form.addRow("Confianza:", self.conf)
        form.addRow("ImgSz:", self.imgsz)
        form.addRow("Process every:", self.pevery)
        form.addRow("Device:", self.device)
        v.addLayout(form)

        # Botonera
        h = QtWidgets.QHBoxLayout()
        self.btn_reload = QtWidgets.QPushButton("🔄 Recargar JSON")
        self.btn_save = QtWidgets.QPushButton("💾 Guardar cambios")
        h.addWidget(self.btn_reload); h.addWidget(self.btn_save)
        v.addLayout(h)

        self.btn_reload.clicked.connect(self.reload_config)
        self.btn_save.clicked.connect(self.save_changes)
        self.cam_combo.currentIndexChanged.connect(self._load_cam_values)

        self.cfg = None
        self.reload_config()

    def reload_config(self):
        try:
            self.cfg = load_config()
            self.json_view.setText(json.dumps(self.cfg, indent=2, ensure_ascii=False))
            # Rellenar combo cámaras
            self.cam_combo.blockSignals(True)
            self.cam_combo.clear()
            for cam in self.cfg.get("cameras", []):
                label = str(cam.get("name") or cam.get("id"))
                self.cam_combo.addItem(label, cam)
            self.cam_combo.blockSignals(False)
            self._load_cam_values()
            if self.reload_cb: self.reload_cb(self.cfg)
        except Exception as e:
            self.json_view.setText(f"Error al cargar config: {e}")

    def _load_cam_values(self):
        if self.cfg is None: return
        y = self.cfg["yolo"]
        self.model.setText(y.get("model",""))
        self.conf.setValue(float(y.get("confidence",0.35)))
        self.imgsz.setValue(int(y.get("imgsz",640)))
        self.pevery.setValue(int(y.get("process_every",1)))
        dev = y.get("device","cpu")
        idx = self.device.findText(dev)
        self.device.setCurrentIndex(idx if idx>=0 else 0)

    def save_changes(self):
        if self.cfg is None: return
        self.cfg["yolo"]["model"] = self.model.text().strip()
        self.cfg["yolo"]["confidence"] = float(self.conf.value())
        self.cfg["yolo"]["imgsz"] = int(self.imgsz.value())
        self.cfg["yolo"]["process_every"] = int(self.pevery.value())
        self.cfg["yolo"]["device"] = self.device.currentText()
        try:
            save_config(self.cfg)
            QtWidgets.QMessageBox.information(self, "Guardado", "Configuración actualizada.")
            self.reload_config()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

# ===================== Cámaras Tab con Combo + Iniciar =====================
class CamerasTab(QtWidgets.QWidget):
    def __init__(self, app_config):
        super().__init__()
        self.cfg = app_config
        self.grid = QtWidgets.QGridLayout(self)

        # Controles superiores
        self.header = QtWidgets.QHBoxLayout()
        self.cam_combo = QtWidgets.QComboBox()
        for cam in self.cfg.get("cameras", []):
            label = str(cam.get("name") or cam.get("id"))
            self.cam_combo.addItem(label, cam)
        self.btn_edit = QtWidgets.QPushButton("⚙️ Editar configuración…")
        self.btn_start = QtWidgets.QPushButton("▶️ Iniciar análisis")
        self.header.addWidget(QtWidgets.QLabel("<b>Fuente:</b>"))
        self.header.addWidget(self.cam_combo)
        self.header.addWidget(self.btn_edit)
        self.header.addWidget(self.btn_start)
        self.grid.addLayout(self.header, 0, 0, 1, 2)

        # Área de cámaras
        self.container = QtWidgets.QWidget()
        self.cam_layout = QtWidgets.QGridLayout(self.container)
        self.grid.addWidget(self.container, 1, 0, 1, 2)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_edit.clicked.connect(self.edit_before_start)
        self.widgets = []  # CameraYoloWidget activos

    def reload_from_config(self, cfg):
        self.cfg = cfg
        self.cam_combo.clear()
        for cam in self.cfg.get("cameras", []):
            label = str(cam.get("name") or cam.get("id"))
            self.cam_combo.addItem(label, cam)

    def edit_before_start(self):
        cam = self.cam_combo.currentData()
        if cam is None: return
        dlg = CameraConfigDialog(self.cfg, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            # guarda cambios globales
            new_cfg = dlg.values()
            save_config(new_cfg)
            self.reload_from_config(load_config())
            QtWidgets.QMessageBox.information(self, "Listo", "Configuración actualizada.")

    def start_camera(self):
        cam = self.cam_combo.currentData()
        if cam is None:
            QtWidgets.QMessageBox.information(self, "Selecciona", "Elige una cámara primero.")
            return
        source = cam.get("id")
        w = CameraYoloWidget(source, load_config())
        # añadir al grid en la primera posición libre
        r = len(self.widgets)//2; c = len(self.widgets)%2
        self.cam_layout.addWidget(w, r, c)
        self.widgets.append(w)

    def close_all(self):
        for w in self.widgets:
            try: w.close()
            except: pass

# ===================== Control Center =====================
class ControlCenter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galu Guard — Control Center v2")
        self.resize(1280, 820)
        self.config = load_config()

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.cameras_tab = CamerasTab(self.config)
        self.dashboard_tab = DashboardTab()
        self.gallery_tab = GalleryTab()
        self.memory_tab = MemoryTab()
        self.config_tab = ConfigTab(self._on_reload_config)

        self.tabs.addTab(self.cameras_tab, "🎥 Cámaras")
        self.tabs.addTab(self.dashboard_tab, "📊 Dashboard")
        self.tabs.addTab(self.gallery_tab, "🖼️ Galería")
        self.tabs.addTab(self.memory_tab, "🧠 Memoria")
        self.tabs.addTab(self.config_tab, "⚙️ Configuración")

    def _on_reload_config(self, cfg):
        self.config = cfg
        self.cameras_tab.reload_from_config(cfg)

    def closeEvent(self, e):
        try: self.cameras_tab.close_all()
        except: pass
        super().closeEvent(e)

# ===================== MAIN =====================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ControlCenter()
    win.show()
    sys.exit(app.exec())
