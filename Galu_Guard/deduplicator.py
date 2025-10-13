# deduplicator.py — Postprocesador para eliminar duplicados en logs/detections.csv
import csv, os, time
from math import hypot
from collections import deque

INPUT_CSV = "logs/detections.csv"
OUTPUT_CSV = "logs/detections_deduped.csv"
DIST_THRESHOLD = 20  # distancia máxima entre centros para considerar duplicado
TIME_WINDOW = 2      # segundos

# Estructura: (timestamp_unix, cam_id, class, x1, y1, x2, y2)
recent_detections = deque()

def box_area(x1, y1, x2, y2):
    return abs((x2 - x1) * (y2 - y1))

def area_similar(a1, a2, tolerance=0.1):
    return abs(a1 - a2) / max(a1, a2) < tolerance


def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_duplicate(ts, cam, cls, x1, y1, x2, y2):
    cx, cy = bbox_center(x1, y1, x2, y2)
    a1 = box_area(x1, y1, x2, y2)
    ts = int(time.mktime(time.strptime(ts, "%Y%m%d_%H%M%S")))
    for old_ts, old_cam, old_cls, ox1, oy1, ox2, oy2 in recent_detections:
        if cam != old_cam or cls != old_cls:
            continue
        if abs(ts - old_ts) > TIME_WINDOW:
            continue
        ocx, ocy = bbox_center(ox1, oy1, ox2, oy2)
        if hypot(cx - ocx, cy - ocy) >= DIST_THRESHOLD:
            continue
        a2 = box_area(ox1, oy1, ox2, oy2)
        if not area_similar(a1, a2):
            continue
        return True
    return False


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

def deduplicate():
    if not os.path.exists(INPUT_CSV):
        print("No se encontró el archivo de entrada.")
        return

    rows = load_csv(INPUT_CSV)
    if not rows: return
    header, data = rows[0], rows[1:]
    deduped = [header]

    for row in data:
        ts, cam_id, cls, conf, x1, y1, x2, y2, img_path = row
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        ts_unix = int(time.mktime(time.strptime(ts, "%Y%m%d_%H%M%S")))

        if not is_duplicate(ts, cam_id, cls, x1, y1, x2, y2):
            deduped.append(row)
            recent_detections.append((ts_unix, cam_id, cls, x1, y1, x2, y2))

        # limpiar viejos
        while recent_detections and ts_unix - recent_detections[0][0] > TIME_WINDOW:
            recent_detections.popleft()

    write_csv(OUTPUT_CSV, deduped)
    print(f"Filtrado completado. {len(deduped)-1} entradas únicas guardadas en {OUTPUT_CSV}")

if __name__ == "__main__":
    deduplicate()
