import cv2
import time

# 🔹 URL del stream RTSP — cámbiala por la IP de tu Raspberry Pi
RTSP_URL = "rtsp://192.168.1.78:8554/cam"  # ejemplo

# Abrimos el stream
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream RTSP.")
    exit()

print("✅ Conectado al stream RTSP. Presiona 'q' para salir.")

cv2.namedWindow("Cámara RTSP - Raspberry Pi", cv2.WINDOW_NORMAL)

while True:
    if not cap.isOpened():
        print("⚠️ Conexión perdida. Reintentando...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(RTSP_URL)
        continue

    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ No se recibió frame del stream. Reintentando...")
        time.sleep(0.5)
        continue

    cv2.imshow("Cámara RTSP - Raspberry Pi", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Saliendo del stream.")
        break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()