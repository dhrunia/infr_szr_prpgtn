FROM tensorflow/tensorflow:2.9.1-gpu
ARG USERNAME=anirudh
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN sudo apt-get update && sudo apt-get install -y git ffmpeg \
    python3-tk wget libfftw3-3 libfftw3-bin libfftw3-dev \
    texlive-xetex texlive-fonts-recommended texlive-plain-generic

RUN pip install --upgrade pip

RUN pip install --upgrade tensorflow-probability==0.17.0 pyshtools==4.10 mne \
    ffmpeg-python matplotlib pylint yapf black ipykernel autopep8 \
    ipympl nbconvert pyvista

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN cd /home/$USERNAME \
    && wget https://bitbucket.org/nschaeff/shtns/downloads/shtns-3.5.2.tar.gz \
    && tar -xzf shtns-3.5.2.tar.gz \
    && cd shtns \
    && ./configure -enable-python \
    && make -j4 \
    && python setup.py install --user
