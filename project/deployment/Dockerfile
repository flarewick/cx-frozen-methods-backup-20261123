FROM nvidia/cuda:12.6.3-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_SHA256
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git pkg-config rsync ffmpeg vulkan-tools \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
    liblzma-dev libncursesw5-dev uuid-dev libgdbm-dev libnss3-dev \
    libgl1 libegl1 libglvnd0 libgles2 libopengl0 libx11-6 libglib2.0-0 libvulkan1 libglfw3 \
    && rm -rf /var/lib/apt/lists/*
RUN test -n "$PYTHON_SHA256" && \
    curl -fL --retry 3 https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz -o /tmp/python.tar.xz && \
    echo "$PYTHON_SHA256  /tmp/python.tar.xz" | sha256sum -c - && \
    tar -xJf /tmp/python.tar.xz -C /tmp && cd /tmp/Python-3.11.9 && \
    LDFLAGS='-Wl,-rpath,/opt/python311/lib' ./configure --prefix=/opt/python311 --enable-shared && \
    make -j4 && make install && rm -rf /tmp/Python-3.11.9 /tmp/python.tar.xz && \
    /opt/python311/bin/python3.11 -m venv /opt/cx_venv
ENV PATH=/opt/cx_venv/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
COPY requirements.runtime.txt environment.json install_overlay.py doctor.py bootstrap.sh /opt/cx_deployment/
ADD runtime.tar.gz /
RUN bash /opt/cx_deployment/bootstrap.sh
WORKDIR /data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA
CMD ["bash"]
