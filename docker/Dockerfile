FROM nvcr.io/nvidia/tensorrt:21.09-py3

ARG UID
ARG GID

# install python package
RUN pip install --upgrade pip && \
    pip install --upgrade opencv_python==4.5.4.58 && \
    pip install --upgrade opencv_python_headless==4.5.4.58 && \
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip install --upgrade timm && \
    pip install --upgrade onnx==1.10.1 && \
    pip install --upgrade onnx_graphsurgeon==0.3.12 --index-url https://pypi.ngc.nvidia.com && \
    pip install --upgrade flake8 black mypy && \
    ldconfig && \
    pip cache purge && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# create user who can sudo
ENV USER dev
ENV GROUP dev
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN groupadd -g ${GID} ${GROUP}
RUN useradd -u ${UID} -g ${GROUP} -m ${USER}
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:dev" | chpasswd
RUN sed -i.bak "s#${HOME}:#${HOME}:${SHELL}#" /etc/passwd

USER ${USER}
WORKDIR ${HOME}
