# launcher.py — Iniciador de Galu Guard
import sys, json, subprocess
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galu Guard Launcher")
        self.setFixedSize(400, 300)

        # --- Título ---
        title = QLabel("Galu Guard Launcher")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 30px;")

        # --- Botones principales ---
        btn_config = QPushButton("Configuración")
        btn_run = QPushButton("Ejecutar cámaras")
        btn_report = QPushButton("Reportador histórico")

        for btn in (btn_config, btn_run, btn_report):
            btn.setFixedHeight(50)

        btn_config.clicked.connect(self.open_config)
        btn_run.clicked.connect(self.run_from_config)
        btn_report.clicked.connect(self.open_reportador)

        # --- Layout ---
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(btn_config)
        layout.addWidget(btn_run)
        layout.addWidget(btn_report)
        self.setLayout(layout)

    # === Abrir configurador ===
    def open_config(self):
        subprocess.Popen(["python", "configurator.py"])

    # === Ejecutar cámaras según config.json ===
    def run_from_config(self):
        try:
            with open("config.json") as f:
                config = json.load(f)
            executor = config.get("executor", "camera_yolo_1.py")
            subprocess.Popen(["python", executor])
        except Exception as e:
            print(f"Error al ejecutar cámara: {e}")

    # === Abrir reportador ===
    def open_reportador(self):
        try:
            subprocess.Popen(["python", "reportador.py"])
        except Exception as e:
            print(f"Error al abrir reportador: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Launcher()
    w.show()
    sys.exit(app.exec())
