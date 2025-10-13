# launcher.py — Iniciador de Galu Guard con selección automática de ejecutor
import sys, json, subprocess
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galu Guard Launcher")
        self.setFixedSize(400, 300)

        title = QLabel("Galu Guard Launcher")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 30px;")

        btn_train = QPushButton("Entrenamiento")
        btn_config = QPushButton("Configuración")
        btn_run = QPushButton("Ejecutar")

        for btn in (btn_train, btn_config, btn_run):
            btn.setFixedHeight(50)

        btn_train.clicked.connect(self.open_training)
        btn_config.clicked.connect(self.open_config)
        btn_run.clicked.connect(self.run_from_config)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(btn_train)
        layout.addWidget(btn_config)
        layout.addWidget(btn_run)
        self.setLayout(layout)

    def open_training(self):
        subprocess.Popen(["python", "trainer.py"])  # Placeholder

    def open_config(self):
        subprocess.Popen(["python", "configurator.py"])

    def run_from_config(self):
        try:
            with open("config.json") as f:
                config = json.load(f)
            executor = config.get("executor", "camera_yolo_1.py")
            subprocess.Popen(["python", executor])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Launcher()
    w.show()
    sys.exit(app.exec())
