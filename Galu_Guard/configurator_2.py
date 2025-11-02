# configurator.py — Configuración dinámica de modelo y clases YOLO
import sys, json, os, glob, cv2
from PySide6.QtWidgets import (QWidget, QApplication, QLabel, QPushButton, QComboBox, QVBoxLayout,
                               QCheckBox, QSpinBox, QScrollArea, QListWidgetItem, QMessageBox,
                               QFileDialog, QLineEdit, QWidget as QtWidget)
from PySide6.QtCore import Qt, QTimer
from ultralytics import YOLO

IS_WIN = os.name == "nt"
BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY


EXCLUIR = {
    "airplane",
    "apple",
    "asus",
    "banana",
    "baseball glove",
    "bear",
    "bed",
    "boat",
    "book",
    "bowl",
    "broccoli",
    "cake",
    "carrot",
    "chair",
    "clock",
    "couch",
    "donut",
    "elephant",
    "fire hydrant",
    "frisbee",
    "giraffe",
    "hot dog",
    "keyboard",
    "kite",
    "laptop",
    "microwave",
    "no-helmet",
    "no-vest",
    "oven",
    "parking meter",
    "pizza",
    "potted plant",
    "remote",
    "refrigerator",
    "sandwich",
    "sheep",
    "sink",
    "skateboard",
    "skis",
    "snowboard",
    "sports ball",
    "stop sign",
    "surfboard",
    "tennis racket",
    "tie",
    "toaster",
    "toothbrush",
    "traffic light",
    "truck",
    "vase",
    "zebra"
}

# Puedes registrar aquí tus modelos disponibles
MODELS_DIR = "models"
MODELS = {
    "YOLO11s (default)": "yolo11s.pt",
    "Casco EPP (helmet.pt)": "runs/detect/train2/weights/best.pt",
    "Otro...": "custom"
}

class Configurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración de Galu Guard")
        self.setFixedSize(500, 650)

        # ---------- Selección de modelo ----------
        self.model_label = QLabel("Seleccionar modelo YOLO:")
        self.model_combo = QComboBox()
        for name in MODELS:
            self.model_combo.addItem(name)
        self.model_combo.currentIndexChanged.connect(self.load_model_and_classes)

        self.model_path = MODELS["YOLO11s (default)"]  # por defecto
        self.custom_model_btn = QPushButton("Seleccionar archivo...")
        self.custom_model_btn.clicked.connect(self.select_custom_model)
        self.custom_model_btn.setVisible(False)

        # ---------- Cámaras ----------
        self.cams_label = QLabel("Seleccionar cámara USB o RTSP:")
        self.cams_combo = QComboBox()
        # self.load_cameras()  # Removed from here

        self.rtsp_label = QLabel("URL RTSP:")
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("Ingrese URL RTSP aquí")
        self.rtsp_input.setReadOnly(False)
        self.rtsp_input.setEnabled(False)
        self.rtsp_input.setClearButtonEnabled(True)
        self.rtsp_input.setFocusPolicy(Qt.StrongFocus)
        self.rtsp_input.setVisible(False)
        self.rtsp_input.setText("rrtsp://10.1.30.186:8554/cam")

        # Add RTSP test connection button below RTSP input
        self.rtsp_test_btn = QPushButton("Probar conexión RTSP")
        self.rtsp_test_btn.setVisible(False)
        self.rtsp_test_btn.clicked.connect(self.test_rtsp_connection)

        # Add RTSP status label below rtsp_input
        self.rtsp_status_label = QLabel("")
        self.rtsp_status_label.setVisible(False)
        self.rtsp_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.load_cameras()  # Inserted here after RTSP widgets creation

        # ---------- Clases dinámicas ----------
        self.obj_label = QLabel("Clases detectables:")
        self.obj_checks = []
        self.scroll_widget = QtWidget()
        self.objects_box = QVBoxLayout(self.scroll_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setFixedHeight(200)

        # Add select all button below the label "Clases detectables:"
        self.select_all_btn = QPushButton("Seleccionar todo")
        self.select_all_btn.clicked.connect(self.select_all_classes)

        # ---------- Parámetros de inferencia ----------
        self.conf_label = QLabel("Confianza:")
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(10, 90)
        self.conf_spin.setValue(35)

        self.imgsz_label = QLabel("Resolución:")
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)

        self.every_label = QLabel("Cada N frames:")
        self.every_spin = QSpinBox()
        self.every_spin.setRange(1, 5)
        self.every_spin.setValue(1)

        self.save_btn = QPushButton("Guardar configuración")
        self.save_btn.clicked.connect(self.save_config)

        self.back_btn = QPushButton("Volver al launcher")
        self.back_btn.clicked.connect(self.return_to_launcher)

        # ---------- Layout general ----------
        layout = QVBoxLayout()
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.custom_model_btn)
        layout.addWidget(self.cams_label)
        layout.addWidget(self.cams_combo)
        layout.addWidget(self.rtsp_label)
        layout.addWidget(self.rtsp_input)
        layout.addWidget(self.rtsp_status_label)
        layout.addWidget(self.rtsp_test_btn)
        layout.addWidget(self.obj_label)
        layout.addWidget(self.select_all_btn)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_spin)
        layout.addWidget(self.imgsz_label)
        layout.addWidget(self.imgsz_spin)
        layout.addWidget(self.every_label)
        layout.addWidget(self.every_spin)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

        self.load_model_and_classes()

    def load_cameras(self):
        self.cams_combo.clear()
        # Add detected USB cameras
        if IS_WIN:
            for i in range(10):
                cap = cv2.VideoCapture(i, BACKEND)
                ok = cap.isOpened()
                if ok: ok, _ = cap.read()
                cap.release()
                if ok:
                    self.cams_combo.addItem(f"Cam {i}", ("USB", i))
        else:
            for path in sorted(glob.glob("/dev/video*")):
                idx = int(path.replace("/dev/video", ""))
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                ok = cap.isOpened()
                if ok: ok, _ = cap.read()
                cap.release()
                if ok:
                    self.cams_combo.addItem(path, ("USB", idx))
        # Add RTSP option
        self.cams_combo.addItem("RTSP", ("RTSP", None))
        self.on_cam_selection_changed(self.cams_combo.currentIndex())

    def on_cam_selection_changed(self, index):
        if index < 0:
            self.rtsp_input.setVisible(False)
            self.rtsp_input.setEnabled(False)
            self.rtsp_input.setReadOnly(True)
            self.rtsp_input.clearFocus()
            self.rtsp_test_btn.setVisible(False)
            self.rtsp_test_btn.setEnabled(False)
            self.rtsp_status_label.setVisible(False)
            self.rtsp_status_label.setEnabled(False)
            return
        data = self.cams_combo.itemData(index)
        if data and data[0] == "RTSP":
            self.rtsp_input.setVisible(True)
            self.rtsp_input.setEnabled(True)
            self.rtsp_input.setReadOnly(False)
            self.rtsp_input.setFocus()
            if not self.rtsp_input.text().strip():
                self.rtsp_input.setText("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
            self.rtsp_test_btn.setVisible(True)
            self.rtsp_test_btn.setEnabled(True)
            self.rtsp_status_label.setVisible(True)
            self.rtsp_status_label.setEnabled(True)
        else:
            self.rtsp_input.setVisible(False)
            self.rtsp_input.setEnabled(False)
            self.rtsp_input.setReadOnly(True)
            self.rtsp_input.clearFocus()
            self.rtsp_test_btn.setVisible(False)
            self.rtsp_test_btn.setEnabled(False)
            self.rtsp_status_label.setVisible(False)
            self.rtsp_status_label.setEnabled(False)

    def select_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo YOLO", ".", "Modelos (*.pt)")
        if path:
            self.model_path = path
            self.load_model_and_classes()

    def load_model_and_classes(self):
        name = self.model_combo.currentText()
        if MODELS[name] == "custom":
            self.custom_model_btn.setVisible(True)
            return  # Espera a que se seleccione
        else:
            self.custom_model_btn.setVisible(False)
            self.model_path = MODELS[name]

        # Cargar modelo
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo cargar el modelo:\n{e}")
            return

        # Limpiar clases anteriores
        for i in reversed(range(self.objects_box.count())):
            self.objects_box.itemAt(i).widget().setParent(None)
        self.obj_checks.clear()

        # Añadir nuevas clases
        for cls in model.names.values():
            if cls.lower() in EXCLUIR:
                continue
            chk = QCheckBox(cls)
            chk.setChecked(False)
            self.obj_checks.append(chk)
            self.objects_box.addWidget(chk)

    def select_all_classes(self):
        for chk in self.obj_checks:
            chk.setChecked(True)

    def test_rtsp_connection(self):
        rtsp_url = self.rtsp_input.text().strip()
        if not rtsp_url:
            QMessageBox.warning(self, "Error", "Por favor ingrese una URL RTSP válida.")
            return

        # Set status label to "Verificando..." and gray color
        self.rtsp_status_label.setText("🔄 Verificando...")
        self.rtsp_status_label.setStyleSheet("color: gray;")
        self.rtsp_status_label.setVisible(True)

        # Use QTimer.singleShot to avoid blocking UI
        def check_stream():
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                self.rtsp_status_label.setText("❌ Inválido")
                self.rtsp_status_label.setStyleSheet("color: red;")
                cap.release()
                return
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.rtsp_status_label.setText("✅ Correcto")
                self.rtsp_status_label.setStyleSheet("color: green;")
            else:
                self.rtsp_status_label.setText("❌ Inválido")
                self.rtsp_status_label.setStyleSheet("color: red;")

        QTimer.singleShot(100, check_stream)

    def save_config(self):
        selected_cams = []
        # Collect USB cameras
        for i in range(self.cams_combo.count()):
            data = self.cams_combo.itemData(i)
            if data and data[0] == "USB":
                # Check if this USB camera is selected (since QComboBox allows one selection, we consider only current)
                # But instruction says both USB and RTSP can coexist, so we must allow multiple selections?
                # Since QComboBox is single selection, we will save only the selected USB camera if any.
                pass
        # Actually, QComboBox single selection, so only one selected item.
        current_index = self.cams_combo.currentIndex()
        if current_index >= 0:
            data = self.cams_combo.itemData(current_index)
            if data:
                if data[0] == "USB":
                    selected_cams.append({
                        "id": data[1],
                        "type": "USB"
                    })
                elif data[0] == "RTSP":
                    rtsp_url = self.rtsp_input.text().strip()
                    if rtsp_url:
                        selected_cams.append({
                            "id": rtsp_url,
                            "type": "RTSP"
                        })

        selected_objs = [chk.text() for chk in self.obj_checks if chk.isChecked()]
        cam_count = len(selected_cams)

        # Determinar archivo ejecutor
        if cam_count == 1:
            executor = "camera_yolo_1.py"
        elif cam_count == 2:
            executor = "camera_yolo_1x2.py"
        elif cam_count <= 4:
            executor = "camera_yolo_2x2.py"
        else:
            executor = "camera_yolo_2x2.py"

        config = {
            "cam_count": cam_count,
            "cameras": selected_cams,
            "executor": executor,
            "objects": selected_objs,
            "yolo": {
                "model": self.model_path,
                "confidence": self.conf_spin.value() / 100,
                "imgsz": self.imgsz_spin.value(),
                "process_every": self.every_spin.value(),
                "device": "cuda"
            }
        }

        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)

        QMessageBox.information(self, "Listo", "Configuración guardada correctamente.")
        self.close()

    def return_to_launcher(self):
        self.close()
        import subprocess
        subprocess.Popen(["python", "launcher.py"])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Configurator()
    w.show()
    sys.exit(app.exec())
