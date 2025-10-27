GALU GUARD – Detección de Objetos en Tiempo Real con Interfaz Qt

Galu Guard es una plataforma de vigilancia visual que combina detección en vivo con YOLOv8, una interfaz gráfica construida con PySide6 (Qt) y funcionalidades avanzadas como configuración visual, reportes históricos y deduplicación. Se ejecuta en un contenedor Docker con soporte GPU mediante NVIDIA Container Toolkit, con acceso a la cámara del host y visualización local vía X11.



Requisitos:

• Ubuntu con GPU NVIDIA y drivers instalados
• Docker
• NVIDIA Container Toolkit
• Entorno gráfico (X11)
• Cámara USB visible como `/dev/video0`



Instalación del NVIDIA Container Toolkit:

1. Agrega el repositorio:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
```

2. Instala los paquetes:

```bash
sudo apt install -y nvidia-container-toolkit
```

3. Configura el runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. Verifica acceso a GPU:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```



Estructura del proyecto:

```bash
Galu_guard_contenedor/
├── camera_yolo_1.py       # Detección y visualización en tiempo real
├── galu_guard_center.py   # Backend principal del sistema
├── launcher.py            # Lanzador gráfico
├── configurator.py        # Configurador visual de cámaras y parámetros
├── reportador.py          # Interfaz de reportes
├── deduplicator.py        # Lógica de deduplicación
├── config.json            # Configuración global
├── dockerfile             # Imagen con PyTorch, Qt y dependencias
├── correr.sh              # Script de ejecución
├── logs/                  # Capturas y registros
```



Construcción de la imagen Docker:

Desde el directorio raíz:

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

Corre el contenedor con:

```bash
./correr.sh
```

Aparecerá la ventana principal Qt, desde la cual puedes lanzar la detección, configurar parámetros, revisar reportes, o deduplicar resultados. El sistema utiliza GPU para acelerar el procesamiento y guarda registros visuales en la carpeta `logs/`.

