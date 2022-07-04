FROM tensorflow/tensorflow:latest-gpu
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

RUN mkdir -p /home/$USERNAME/.ssh
# COPY .ssh/git_rsa /home/$USERNAME/.ssh/id_rsa
# RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

RUN apt-get update && apt-get install -y \
    openssh-client \
    git

RUN pip install --upgrade tensorflow-probability pyshtools ffmpeg-python \
    matplotlib

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME