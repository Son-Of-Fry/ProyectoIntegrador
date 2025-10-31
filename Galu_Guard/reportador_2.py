import sys, csv, os, glob, time
from PySide6 import QtCore, QtGui, QtWidgets
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

CSV_PATH = "logs/detections_deduped.csv"
IMAGES_DIR = "logs/images"
RELATIONS_PATH = "logs/relations.csv"
REFRESH_INTERVAL = 5000  # ms
SYNC_REMOVE_MISSING = True  # elimina del CSV las filas cuyo archivo ya no existe


class Reportador(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reportador histórico - Galu Guard")
        self.resize(950, 850)

        # === Tabla de resumen ===
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Clase", "Cámara", "Conteo"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # === Gráfico de barras ===
        self.chart_canvas = plt.figure(figsize=(8,3))
        self.chart_widget = QtWidgets.QWidget()
        self.chart_layout = QtWidgets.QVBoxLayout(self.chart_widget)
        self.chart_canvas_canvas = plt.backends.backend_qt5agg.FigureCanvasQTAgg(self.chart_canvas)
        self.chart_layout.addWidget(self.chart_canvas_canvas)

        # === Log de eventos ===
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color:#111;color:#0f0;font-family:monospace;")

        # === Galería (desde disco) ===
        self.gallery = QtWidgets.QListWidget()
        self.gallery.setViewMode(QtWidgets.QListView.IconMode)
        self.gallery.setIconSize(QtCore.QSize(160, 120))
        self.gallery.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.gallery.setMovement(QtWidgets.QListWidget.Static)
        self.gallery.setSpacing(10)
        self.gallery.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.gallery.itemSelectionChanged.connect(self.show_image_detail)

        # === Panel de detalle ===
        self.detail_label = QtWidgets.QLabel("Selecciona una o varias imágenes para ver detalles")
        self.detail_label.setAlignment(QtCore.Qt.AlignLeft)

        # === Botones ===
        self.delete_btn = QtWidgets.QPushButton("🗑️ Eliminar imagen seleccionada")
        self.delete_btn.clicked.connect(self.delete_selected_images)

        self.same_btn = QtWidgets.QPushButton("🧩 Marcar como misma persona")
        self.same_btn.clicked.connect(self.mark_same_person)

        # === Layout general ===
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Resumen histórico de detecciones</b>"))
        layout.addWidget(self.table)
        layout.addWidget(self.chart_widget)
        layout.addWidget(QtWidgets.QLabel("<b>Últimos eventos</b>"))
        layout.addWidget(self.log_box)
        layout.addWidget(QtWidgets.QLabel("<b>Galería de imágenes (archivos reales)</b>"))
        layout.addWidget(self.gallery)
        layout.addWidget(self.detail_label)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.delete_btn)
        btns.addWidget(self.same_btn)
        layout.addLayout(btns)

        # Estado
        self.csv_index = {}   # path -> (ts, cam, cls, obj_id, conf, x1,y1,x2,y2)
        self.current_selection = []

        # Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_report)
        self.timer.start(REFRESH_INTERVAL)

        self.update_report()

    # ===================== Carga CSV =====================
    def load_csv(self):
        """Carga CSV completo y opcionalmente limpia entradas con archivos faltantes."""
        if not os.path.exists(CSV_PATH):
            return []

        with open(CSV_PATH, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) <= 1:
            return []

        header, data = rows[0], rows[1:]
        # Reconstruir índice y (opcional) limpiar faltantes
        cleaned = [header]
        self.csv_index.clear()

        removed = 0
        for r in data:
            if len(r) < 10:
                continue
            ts, cam, cls, obj_id, conf, x1, y1, x2, y2, img_path = r
            if os.path.exists(img_path):
                cleaned.append(r)
                self.csv_index[img_path] = (ts, cam, cls, obj_id, conf, x1, y1, x2, y2)
            else:
                removed += 1

        # Si activado, guardar CSV sin rutas rotas
        if SYNC_REMOVE_MISSING and removed > 0:
            try:
                with open(CSV_PATH, "w", newline="") as f:
                    csv.writer(f).writerows(cleaned)
            except Exception as e:
                print("No se pudo sincronizar CSV:", e)

        return cleaned[1:]  # solo data

    # ===================== Actualizar reportes =====================
    def update_report(self):
        rows = self.load_csv()

        # === Resumen y eventos desde CSV ===
        conteo = defaultdict(Counter)
        ultimos = []
        for r in rows:
            try:
                ts, cam, cls, obj_id, conf, *_ = r
                conteo[cls][cam] += 1
                ultimos.append(f"[{ts}] Cam{cam} → {cls} (ID {obj_id}, {conf})")
            except Exception:
                continue

        # Tabla
        self.table.setRowCount(0)
        for cls, cams in conteo.items():
            for cam, count in cams.items():
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(cls))
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(cam)))
                self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(count)))

        # Gráfico de barras con conteo total por clase
        self.chart_canvas.clear()
        ax = self.chart_canvas.add_subplot(111)
        totals = {cls: sum(cams.values()) for cls, cams in conteo.items()}
        classes = list(totals.keys())
        counts = list(totals.values())
        ax.bar(classes, counts)
        ax.set_xlabel('Clase')
        ax.set_ylabel('Conteo total')
        ax.set_title('Conteo total por clase')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        self.chart_canvas_canvas.draw()

        # Últimos eventos
        self.log_box.setText("\n".join(reversed(ultimos[-30:])))

        # === Galería desde disco ===
        self.load_gallery_from_disk()

    # ===================== Galería desde disco =====================
    def load_gallery_from_disk(self):
        self.gallery.clear()
        os.makedirs(IMAGES_DIR, exist_ok=True)

        # Lista todas las imágenes reales
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        # Orden cronológico descendente (más nuevas primero)
        files.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)

        for path in files:
            item = QtWidgets.QListWidgetItem()
            # Icono
            try:
                icon = QtGui.QIcon(path)
            except Exception:
                icon = QtGui.QIcon.fromTheme("image-missing")
            item.setIcon(icon)

            # Etiqueta: si está en CSV, usa clase y cámara; si no, “unknown”
            if path in self.csv_index:
                ts, cam, cls, obj_id, conf, *_ = self.csv_index[path]
                item.setText(f"{cls}\nCam{cam}")
            else:
                item.setText("unknown\nCam?")
            item.setToolTip(path)
            # Guarda metadata si existe
            meta = self.csv_index.get(path, None)
            item.setData(QtCore.Qt.UserRole, meta)
            self.gallery.addItem(item)

    # ===================== Detalle =====================
    def show_image_detail(self):
        selected = self.gallery.selectedItems()
        self.current_selection = selected

        if not selected:
            self.detail_label.setText("Selecciona una o varias imágenes para ver detalles")
            return

        if len(selected) == 1:
            path = selected[0].toolTip()
            meta = selected[0].data(QtCore.Qt.UserRole)
            if meta:
                ts, cam, cls, obj_id, conf, x1, y1, x2, y2 = meta
                fecha = f"{ts[:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
                self.detail_label.setText(
                    f"<b>Clase:</b> {cls}  |  <b>Cámara:</b> {cam}  |  <b>ID:</b> {obj_id}  |  "
                    f"<b>Conf:</b> {conf}<br><b>Fecha/Hora:</b> {fecha}<br><b>Ruta:</b> {path}"
                )
            else:
                # sin metadata: mostrar mtime
                mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
                self.detail_label.setText(f"<b>Ruta:</b> {path}<br><b>Modificado:</b> {mtime}")
        else:
            self.detail_label.setText(f"{len(selected)} imágenes seleccionadas")

    # ===================== Eliminar imágenes =====================
    def delete_selected_images(self):
        selected = self.gallery.selectedItems()
        if not selected:
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Confirmar eliminación",
            f"¿Eliminar {len(selected)} imagen(es) y sus registros del CSV (si existen)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        # Cargar CSV actual
        rows = self.load_csv()
        header = ["timestamp","cam_id","class","id","conf","x1","y1","x2","y2","image_path"]

        # Conjunto de rutas a eliminar
        to_delete = {item.toolTip() for item in selected}

        # Borra archivos
        for p in to_delete:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                print("Error al borrar", p, e)

        # Reescribe CSV sin esas rutas
        filtered = [r for r in rows if len(r) >= 10 and r[9] not in to_delete]
        try:
            with open(CSV_PATH, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(filtered)
        except Exception as e:
            print("No se pudo actualizar CSV:", e)

        # Refrescar vistas
        self.update_report()

    # ===================== Marcar misma persona =====================
    def mark_same_person(self):
        selected = self.gallery.selectedItems()
        if len(selected) < 2:
            QtWidgets.QMessageBox.information(self, "Información", "Selecciona al menos dos imágenes.")
            return

        os.makedirs(os.path.dirname(RELATIONS_PATH), exist_ok=True)
        with open(RELATIONS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    img1 = selected[i].toolTip()
                    img2 = selected[j].toolTip()
                    writer.writerow([img1, img2, "same"])

        QtWidgets.QMessageBox.information(self, "Guardado", "Relaciones registradas. Se usarán en la próxima ejecución de la cámara.")
        # refresco opcional
        # self.update_report()


# ===================== MAIN =====================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Reportador()
    win.show()
    app.exec()
