# deduplicator.py — Postprocesador para eliminar duplicados en logs/detections.csv
import csv, os, time
from math import hypot
from collections import deque

INPUT_CSV = "logs/detections.csv"
OUTPUT_CSV = "logs/detections_deduped.csv"
DIST_THRESHOLD = 20   # Distancia máxima entre centros para considerar duplicado
TIME_WINDOW = 2       # Segundos para agrupar detecciones cercanas

# Estructura: (timestamp_unix, cam_id, class, x1, y1, x2, y2)
recent_detections = deque()

def box_area(x1, y1, x2, y2):
    return abs((x2 - x1) * (y2 - y1))

def area_similar(a1, a2, tolerance=0.1):
    return abs(a1 - a2) / max(a1, a2) < tolerance

def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_duplicate(ts, cam, cls, x1, y1, x2, y2):
    """Compara detección actual con las recientes para ver si ya existe."""
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
    """Filtra detecciones duplicadas y elimina imágenes redundantes."""
    if not os.path.exists(INPUT_CSV):
        print("No se encontró el archivo de entrada.")
        return

    rows = load_csv(INPUT_CSV)
    if not rows:
        print("El CSV está vacío.")
        return

    header, data = rows[0], rows[1:]
    deduped = [header]
    deleted_count = 0

    for row in data:
        # Se ajusta a 10 columnas incluyendo el ID
        ts, cam_id, cls, obj_id, conf, x1, y1, x2, y2, img_path = row
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        ts_unix = int(time.mktime(time.strptime(ts, "%Y%m%d_%H%M%S")))

        # Verifica si es duplicado según cercanía espacial y temporal
        if not is_duplicate(ts, cam_id, cls, x1, y1, x2, y2):
            deduped.append(row)
            recent_detections.append((ts_unix, cam_id, cls, x1, y1, x2, y2))
        else:
            # Borra imagen asociada si existe
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"No se pudo borrar {img_path}: {e}")

        # Limpieza de detecciones viejas
        while recent_detections and ts_unix - recent_detections[0][0] > TIME_WINDOW:
            recent_detections.popleft()

    write_csv(OUTPUT_CSV, deduped)
    print(f"Filtrado completado. {len(deduped)-1} entradas únicas guardadas en {OUTPUT_CSV}")
    print(f"Se eliminaron {deleted_count} imágenes duplicadas.")

if __name__ == "__main__":
    deduplicate()
