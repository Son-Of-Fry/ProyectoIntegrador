import sys, cv2, time
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import solutions

# ===================== HILO DE PROCESAMIENTO =====================
class YOLOWorker(QtCore.QObject):
    """Hilo que procesa frames con YOLO sin bloquear la GUI."""
    result_ready = QtCore.Signal(object)
    process_request = QtCore.Signal(object)  # recibe frames desde la GUI

    def __init__(self, region_points, model_path="yolo11n.pt"):
        super().__init__()
        self.region_points = region_points
        self.counter = solutions.ObjectCounter(
            region=region_points,
            model=model_path,
            show=False  # sin cv2.imshow
        )
        self.running = True
        self.process_request.connect(self.process_frame)

    @QtCore.Slot(object)
    def process_frame(self, frame):
        if not self.running:
            return
        results = self.counter(frame)
        self.result_ready.emit(results.plot_im)


# ===================== INTERFAZ PRINCIPAL =====================
class YoloCountApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO11 Object Counting - Qt")
        self.resize(900, 600)

        # ---------- Layout ----------
        layout = QtWidgets.QVBoxLayout(self)
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # ---------- Cámara ----------
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("❌ No se pudo abrir la cámara (prueba índice 1).")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.region_points = [
            (100, h // 2 + 30),
            (w - 100, h // 2 + 30),
            (w - 100, h // 2 - 30),
            (100, h // 2 - 30)
        ]

        # ---------- Hilo YOLO ----------
        self.worker = YOLOWorker(self.region_points)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.worker.result_ready.connect(self.update_image)
        self.thread.start()

        # ---------- Timer de captura ----------
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(30)

        # ---------- Etiqueta de conteo ----------
        self.label_count = QtWidgets.QLabel("Conteo total: 0", alignment=QtCore.Qt.AlignCenter)
        self.label_count.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: white; background-color: #222; padding: 5px;"
        )
        layout.addWidget(self.label_count)
        self.total_count = 0

    # ===================== CAPTURA =====================
    def capture_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print("⚠️ No se pudo leer frame de la cámara.")
            return
        self.current_frame = frame
        # Emite la señal con el frame al hilo
        self.worker.process_request.emit(frame)

    # ===================== ACTUALIZA INTERFAZ =====================
    def update_ui(self, frame, tracks):
        # ====== Actualiza la imagen del video ======
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

        # ====== Actualiza tabla (modo histórico) ======
        for t in tracks:
            track_id = str(t["id"])
            existing_row = None

            # Buscar si el ID ya existe en la tabla
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item and item.text() == track_id:
                    existing_row = row
                    break

            if existing_row is not None:
                # Si ya existe, solo actualiza los valores
                self.table.setItem(existing_row, 1, QtWidgets.QTableWidgetItem(t["cls"]))
                self.table.setItem(existing_row, 2, QtWidgets.QTableWidgetItem(f"{t['conf']:.2f}"))
                self.table.setItem(existing_row, 3, QtWidgets.QTableWidgetItem(t["fecha"]))
                self.table.setItem(existing_row, 4, QtWidgets.QTableWidgetItem(t["hora"]))
            else:
                # Si no existe, agrega una nueva fila
                new_row = self.table.rowCount()
                self.table.insertRow(new_row)
                self.table.setItem(new_row, 0, QtWidgets.QTableWidgetItem(track_id))
                self.table.setItem(new_row, 1, QtWidgets.QTableWidgetItem(t["cls"]))
                self.table.setItem(new_row, 2, QtWidgets.QTableWidgetItem(f"{t['conf']:.2f}"))
                self.table.setItem(new_row, 3, QtWidgets.QTableWidgetItem(t["fecha"]))
                self.table.setItem(new_row, 4, QtWidgets.QTableWidgetItem(t["hora"]))

        # ====== Guarda tracks recientes (para CSV) ======
        if tracks:
            # Actualiza o reemplaza los registros previos del mismo ID
            for t in tracks:
                self.report_data = [r for r in self.report_data if r["id"] != t["id"]]
                self.report_data.append(t)

        # ====== Actualiza info de FPS y conteo ======
        now = time.time()
        fps = 1.0 / (now - self.last_time) if (now - self.last_time) > 0 else 0
        self.last_time = now
        active_ids = [t["id"] for t in tracks]
        self.info_label.setText(
            f"FPS: {fps:.1f} | Tracks activos: {len(active_ids)} | "
            f"Total detectados: {self.table.rowCount()} | "
            f"Último reporte: {self.last_report_name}"
        )

    # ===================== LIMPIEZA =====================
    def closeEvent(self, event):
        self.timer.stop()
        self.worker.running = False
        self.thread.quit(); self.thread.wait()
        if self.cap:
            self.cap.release()
        event.accept()


# ===================== MAIN =====================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = YoloCountApp()
    win.show()
    sys.exit(app.exec())
