# configurator.py — Configuración dinámica de modelo y clases YOLO
import sys, json, os, glob, cv2
from PySide6.QtWidgets import (QWidget, QApplication, QLabel, QPushButton, QComboBox, QVBoxLayout,
                               QCheckBox, QSpinBox, QScrollArea, QListWidgetItem, QMessageBox,
                               QFileDialog, QLineEdit, QWidget as QtWidget, QHBoxLayout, QSizePolicy)
from PySide6.QtCore import Qt, QTimer
from ultralytics import YOLO
import subprocess

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

# Intentar cargar modelos desde 'modelos_disponibles.json', si no existe usar valores por defecto
MODELS = {}
MODELS_PATH = os.path.join(os.path.dirname(__file__), "modelos_disponibles.json")
if os.path.exists(MODELS_PATH):
    try:
        with open(MODELS_PATH, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "modelos" in data:
                MODELS = {item["nombre"]: item["archivo"] for item in data["modelos"] if "nombre" in item and "archivo" in item}
    except Exception as e:
        print("Error al cargar modelos_disponibles.json:", e)

class Configurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración de Galu Guard")
        self.setFixedSize(800, 650)  # Aumentado ancho a 800

        # ---------- Selección de modelo ----------
        self.model_label = QLabel("Seleccionar modelo YOLO:")
        self.model_combo = QComboBox()
        for name in MODELS:
            self.model_combo.addItem(name)
        self.model_combo.currentIndexChanged.connect(self.load_model_and_classes)

        self.model_path = None
        # Establecer model_path inicial según primer modelo
        first_model_name = self.model_combo.itemText(0)
        if first_model_name in MODELS and MODELS[first_model_name] != "custom":
            self.model_path = MODELS[first_model_name]
        self.custom_model_btn = QPushButton("Seleccionar archivo...")
        self.custom_model_btn.clicked.connect(self.select_custom_model)
        self.custom_model_btn.setVisible(False)

        # ---------- Selección de tipo de cámara ----------
        self.cam_type_label = QLabel("Seleccionar tipo de cámara:")
        self.cam_type_combo = QComboBox()
        self.cam_type_combo.addItems(["USB", "RTSP", "REALSENSE"])
        self.cam_type_combo.currentIndexChanged.connect(self.on_cam_type_changed)

        # ---------- Selección de dispositivo según tipo ----------
        self.cam_device_label = QLabel("Seleccionar dispositivo:")
        self.cam_device_combo = QComboBox()
        self.cam_device_combo.currentIndexChanged.connect(self.on_cam_device_changed)

        # RTSP widgets (campo URL y botón probar)
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("Ingrese URL RTSP aquí")
        self.rtsp_input.setClearButtonEnabled(True)
        self.rtsp_input.setVisible(False)
        self.rtsp_input.setEnabled(False)
        self.rtsp_test_btn = QPushButton("Probar conexión RTSP")
        self.rtsp_test_btn.setVisible(False)
        self.rtsp_test_btn.clicked.connect(self.test_rtsp_connection)
        self.rtsp_status_label = QLabel("")
        self.rtsp_status_label.setVisible(False)
        self.rtsp_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # REALSENSE info label
        self.realsense_info_label = QLabel("Opciones REALSENSE: 'default'")
        self.realsense_info_label.setVisible(False)

        # ---------- Clases dinámicas ----------
        self.obj_label = QLabel("Clases detectables:")
        self.obj_checks = []
        self.scroll_widget = QtWidget()
        self.objects_box = QVBoxLayout(self.scroll_widget)
        self.objects_box.setContentsMargins(20, 0, 0, 0)  # Márgenes para mover checkboxes a la derecha

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setFixedHeight(200)

        # Botones seleccionar/deseleccionar todo en fila al final del listado
        self.select_all_btn = QPushButton("Seleccionar todo")
        self.select_all_btn.setFixedWidth(120)
        self.select_all_btn.setFixedHeight(25)
        self.select_all_btn.clicked.connect(self.select_all_classes)

        self.deselect_all_btn = QPushButton("Deseleccionar todo")
        self.deselect_all_btn.setFixedWidth(120)
        self.deselect_all_btn.setFixedHeight(25)
        self.deselect_all_btn.clicked.connect(self.deselect_all_classes)

        select_buttons_layout = QHBoxLayout()
        select_buttons_layout.addWidget(self.select_all_btn)
        select_buttons_layout.addWidget(self.deselect_all_btn)
        select_buttons_layout.addStretch()

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

        # ---------- Layout general reorganizado en dos columnas ----------
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.model_combo)
        left_layout.addWidget(self.custom_model_btn)

        left_layout.addWidget(self.cam_type_label)
        left_layout.addWidget(self.cam_type_combo)
        left_layout.addWidget(self.cam_device_label)
        left_layout.addWidget(self.cam_device_combo)

        left_layout.addWidget(self.rtsp_input)
        left_layout.addWidget(self.rtsp_status_label)
        left_layout.addWidget(self.rtsp_test_btn)

        left_layout.addWidget(self.realsense_info_label)

        left_layout.addWidget(self.conf_label)
        left_layout.addWidget(self.conf_spin)
        left_layout.addWidget(self.imgsz_label)
        left_layout.addWidget(self.imgsz_spin)
        left_layout.addWidget(self.every_label)
        left_layout.addWidget(self.every_spin)
        left_layout.addWidget(self.save_btn)
        left_layout.addWidget(self.back_btn)
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.obj_label)
        right_layout.addWidget(self.scroll_area)
        right_layout.addLayout(select_buttons_layout)
        right_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

        self.load_model_and_classes()
        self.on_cam_type_changed(0)  # Inicializar lista dispositivos USB

    def load_cameras_usb(self):
        devices = []
        if IS_WIN:
            for i in range(10):
                cap = cv2.VideoCapture(i, BACKEND)
                ok = cap.isOpened()
                if ok: ok, _ = cap.read()
                cap.release()
                if ok:
                    devices.append(i)
        else:
            for path in sorted(glob.glob("/dev/video*")):
                try:
                    idx = int(path.replace("/dev/video", ""))
                except Exception:
                    continue
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                ok = cap.isOpened()
                if ok: ok, _ = cap.read()
                cap.release()
                if ok:
                    devices.append(idx)
        return devices

    def on_cam_type_changed(self, index):
        cam_type = self.cam_type_combo.currentText()
        self.rtsp_input.setVisible(False)
        self.rtsp_input.setEnabled(False)
        self.rtsp_test_btn.setVisible(False)
        self.rtsp_status_label.setVisible(False)
        self.realsense_info_label.setVisible(False)
        self.cam_device_combo.setVisible(True)
        self.cam_device_combo.clear()

        if cam_type == "USB":
            devices = self.load_cameras_usb()
            if devices:
                for d in devices:
                    self.cam_device_combo.addItem(f"Cam {d}", d)
            else:
                self.cam_device_combo.addItem("No se detectaron cámaras USB", None)
                self.cam_device_combo.setEnabled(False)
            self.cam_device_combo.setEnabled(True)
        elif cam_type == "RTSP":
            self.cam_device_combo.setVisible(False)
            self.rtsp_input.setVisible(True)
            self.rtsp_input.setEnabled(True)
            self.rtsp_input.setText("rtsp://")
            self.rtsp_test_btn.setVisible(True)
            self.rtsp_status_label.setVisible(True)
        elif cam_type == "REALSENSE":
            self.cam_device_combo.setVisible(False)
            self.realsense_info_label.setVisible(True)

    def on_cam_device_changed(self, index):
        # No acciones necesarias por ahora
        pass

    def select_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo YOLO", ".", "Modelos (*.pt)")
        if path:
            self.model_path = path
            self.load_model_and_classes()

    def load_model_and_classes(self):
        name = self.model_combo.currentText()
        if name not in MODELS:
            self.custom_model_btn.setVisible(False)
            self.model_path = None
            return
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
            w = self.objects_box.itemAt(i).widget()
            if w:
                w.setParent(None)
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

    def deselect_all_classes(self):
        for chk in self.obj_checks:
            chk.setChecked(False)

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
        cam_type = self.cam_type_combo.currentText()
        if cam_type == "USB":
            idx = self.cam_device_combo.currentData()
            if idx is not None:
                selected_cams.append({
                    "id": idx,
                    "type": "USB"
                })
        elif cam_type == "RTSP":
            rtsp_url = self.rtsp_input.text().strip()
            if rtsp_url:
                selected_cams.append({
                    "id": rtsp_url,
                    "type": "RTSP"
                })
        elif cam_type == "REALSENSE":
            selected_cams.append({
                "id": "default",
                "type": "REALSENSE"
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

        # Ejecutar el script correspondiente sin cerrar la ventana
        try:
            subprocess.Popen(["python", executor])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo ejecutar el script:\n{e}")

    def return_to_launcher(self):
        self.close()
        subprocess.Popen(["python", "launcher.py"])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Configurator()
    w.show()
    sys.exit(app.exec())
