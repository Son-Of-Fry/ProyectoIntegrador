# configurator.py — Configuración dinámica de modelo y clases YOLO
import sys, json, os, glob, cv2
from PySide6.QtWidgets import (QWidget, QApplication, QLabel, QPushButton, QListWidget, QVBoxLayout,
                               QCheckBox, QSpinBox, QScrollArea, QListWidgetItem, QMessageBox,
                               QFileDialog, QComboBox, QWidget as QtWidget)
from PySide6.QtCore import Qt
from ultralytics import YOLO

IS_WIN = os.name == "nt"
BACKEND = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY


EXCLUIR = {"no-vest","no-helmet","broccoli", 
"asus", 
"toothbrush",
    "airplane",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "tie",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "laptop",
    "remote",
    "keyboard",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase"
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
        self.cams_label = QLabel("Seleccionar cámaras disponibles:")
        self.cams_list = QListWidget()
        self.load_cameras()

        # ---------- Clases dinámicas ----------
        self.obj_label = QLabel("Clases detectables:")
        self.obj_checks = []
        self.scroll_widget = QtWidget()
        self.objects_box = QVBoxLayout(self.scroll_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setFixedHeight(200)

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
        layout.addWidget(self.cams_list)
        layout.addWidget(self.obj_label)
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
        self.cams_list.clear()
        if IS_WIN:
            for i in range(10):
                cap = cv2.VideoCapture(i, BACKEND)
                ok = cap.isOpened()
                if ok: ok, _ = cap.read()
                cap.release()
                if ok:
                    item = QListWidgetItem(f"Cam {i}")
                    item.setCheckState(Qt.Checked)
                    item.setData(Qt.UserRole, i)
                    self.cams_list.addItem(item)
        else:
            for path in sorted(glob.glob("/dev/video*")):
                idx = int(path.replace("/dev/video", ""))
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                ok = cap.isOpened()
                if ok: ok, _ = cap.read()
                cap.release()
                if ok:
                    item = QListWidgetItem(path)
                    item.setCheckState(Qt.Checked)
                    item.setData(Qt.UserRole, idx)
                    self.cams_list.addItem(item)

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
            chk.setChecked(True)
            self.obj_checks.append(chk)
            self.objects_box.addWidget(chk)

    def save_config(self):
        selected_cams = []
        for i in range(self.cams_list.count()):
            item = self.cams_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_cams.append({
                    "id": item.data(Qt.UserRole),
                    "type": "USB"
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
