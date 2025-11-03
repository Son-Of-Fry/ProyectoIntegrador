#!/usr/bin/env python3
"""
camera_realsense_qt.py — Visualizador Qt para cámara Intel RealSense
Usa PySide6 + pyrealsense2 para mostrar el feed RGB en tiempo real.
"""

import sys, time
import numpy as np
import cv2
import pyrealsense2 as rs
from PySide6 import QtCore, QtGui, QtWidgets


class RealSenseViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intel RealSense Viewer (RGB)")
        self.resize(800, 600)

        # QLabel para el video
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)

        # Inicializa RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        print("✅ Cámara RealSense iniciada")

        self.prev_time = time.time()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    def update_frame(self):
        """Captura y actualiza el frame RGB en pantalla"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())

        # Calcula FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convierte BGR → RGB → QImage
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        """Detiene cámara y limpia"""
        print("🛑 Cerrando cámara RealSense...")
        self.timer.stop()
        self.pipeline.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = RealSenseViewer()
    viewer.show()
    app.exec()
