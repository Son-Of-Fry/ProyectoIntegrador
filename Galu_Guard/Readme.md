# GALU GUARD – Detección de Objetos en Tiempo Real con Interfaz Qt

Galu Guard es una plataforma de vigilancia visual que combina detección en vivo con YOLOv8, una interfaz gráfica construida con PySide6 (Qt) y funcionalidades avanzadas de configuración, reporte histórico y deduplicación. El sistema corre completamente dentro de un contenedor Docker con soporte para aceleración GPU (CUDA) y acceso a cámaras USB.



Requisitos:

• Ubuntu con GPU NVIDIA y drivers compatibles
• Docker instalado y funcionando
• NVIDIA Container Toolkit (`nvidia-ctk runtime configure`)
• Sistema gráfico X11 funcionando en el host
• Cámara accesible vía `/dev/video0`


Estructura del proyecto:

```bash
Galu_guard_contenedor/
├── camera_yolo_1.py       # Lógica principal de captura, detección y visualización
├── galu_guard_center.py   # Módulo auxiliar (procesamiento, backend)
├── launcher.py            # Lanzador gráfico con opciones de ejecución
├── configurator.py        # Configuración GUI de cámaras y parámetros
├── reportador.py          # Reporte histórico con imágenes y estadísticas
├── deduplicator.py        # Limpieza de duplicados en detecciones
├── config.json            # Archivo central de configuración
├── dockerfile             # Imagen base con todo preinstalado
├── correr.sh              # Script para lanzar el contenedor de forma segura
├── logs/                  # Carpeta donde se guardan detecciones e imágenes
```



Construcción de la imagen Docker:

Desde el directorio principal:

```bash
docker build -t qt-yolo .
```


Script de ejecución (`correr.sh`):

```bash
#!/bin/bash

docker stop qt-yolo 2>/dev/null
docker rm qt-yolo 2>/dev/null

xhost +local:docker

docker run -it --rm \
  --name qt-yolo \
  --gpus all \
  --device /dev/video0:/dev/video0 \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  qt-yolo

xhost -local:docker
```

Hazlo ejecutable:

```bash
chmod +x correr.sh
```



Ejecutar el sistema:

Simplemente corre:

```bash
./correr.sh
```

Esto iniciará el contenedor con soporte gráfico, GPU y cámara. Se abrirá la ventana del lanzador Qt, desde donde puedes:
• Configurar parámetros (`configurator.py`)
• Ejecutar cámaras (`camera_yolo_1.py`)
• Visualizar reportes históricos (`reportador.py`)

Todo basado en lo definido en `config.json`, que permite personalizar cámaras, clases a detectar, modelo y parámetros de YOLOv8. El sistema guarda logs y genera imágenes organizadas que luego puedes revisar desde la interfaz.

