# ---- CUDA + Ubuntu base (GPU runtime) ----
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /opt

# ---- System deps + Python 3.12 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates git build-essential \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# ---- uv (Python environment manager) ----
RUN python3.12 -m ensurepip --upgrade || true \
 && python3.12 -m pip install --no-cache-dir --upgrade pip \
 && python3.12 -m pip install --no-cache-dir uv \
 && uv --version
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.12

# ---- Clone QiliSDK at v0.1.6 ----
ARG GIT_REF=0.1.6
RUN git clone --depth=1 --branch ${GIT_REF} https://github.com/qilimanjaro-tech/qilisdk.git /opt/qilisdk

# ---- Sync environment from uv.lock + extras ----
WORKDIR /opt/qilisdk
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cuda --extra qutip --extra speqtrum

# Put the project venv on PATH
ENV PATH="/opt/qilisdk/.venv/bin:${PATH}"

# ---- JupyterLab installation ----
RUN uv pip install jupyterlab ipykernel && \ 
	python -m ipykernel install --name qilisdk --display-name "Python 3.12 (qilisdk)" --sys-prefix

# ---- Expose and run JupyterLab ----
WORKDIR /work
RUN mkdir -p /work

EXPOSE 8000
CMD ["jupyter","lab", \
     "--ip=0.0.0.0", \
     "--port=8000",  \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.root_dir=/work", \
     "--NotebookApp.default_url=/lab/tree/work"] \
