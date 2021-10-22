FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ARG UID
ARG GID

# sudo権限を持つ一般ユーザーを作成
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
