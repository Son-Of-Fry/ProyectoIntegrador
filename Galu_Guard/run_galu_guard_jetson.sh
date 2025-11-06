#!/bin/bash
# ===============================================================
# 🚀 Galu Guard Jetson — ejecución Docker con soporte Qt + cámara
# ===============================================================

# Detectar display actual (útil si vienes por SSH o Jetson local)
export DISPLAY=${DISPLAY:-:0}

# Verificar si el display está disponible
if [ -z "$DISPLAY" ] || [ "$DISPLAY" == "" ]; then
  echo "❌ No hay display activo (X11). Conecta monitor o usa 'export DISPLAY=:0'"
  exit 1
fi

# Permitir que Docker acceda al entorno gráfico
xhost +local:root

# Detener y limpiar instancias previas
docker stop galu-guard-jetson 2>/dev/null
docker rm galu-guard-jetson 2>/dev/null

# Ejecutar contenedor con entorno gráfico
sudo docker run -it --rm \
    --name galu-guard-jetson \
    --runtime nvidia \
    --privileged \
    --network host \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/app \
    --device /dev:/dev \
    galu_guard_jetson   # 👈 ya en la misma línea

# Revocar acceso gráfico
xhost -local:root