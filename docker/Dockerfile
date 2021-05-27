FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-dev \
    python3-opencv \
    ninja-build \
    cmake \
    sudo \
    wget \
    vim \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python

# Create a working directory
RUN mkdir /workspace
WORKDIR /workspace

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /workspace
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install pip
ENV PATH="/home/user/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
    pip install --upgrade pip && \
	rm get-pip.py

# Install main packages
RUN pip install --user torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/cu101/torch_stable.html

# nvidia packages
RUN pip install nvidia-pyindex
RUN pip install tritonclient[all]
RUN pip install nvidia-dali-cuda100 --extra-index-url https://developer.download.nvidia.com/compute/redist

# Setup CUDA paths
ENV CUDA_HOME="/usr/local/cuda-10.1"
ENV CUDA_PATH="/usr/local/cuda-10.1"
ENV LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV FORCE_CUDA="1"

# Install and complile Mish-Cuda Activation Function
RUN git clone https://github.com/ioangatop/mish-cuda /home/user/mc &&\
    cd /home/user/mc &&\
    sudo TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing" python setup.py build install &&\
    sudo mv ./build/lib.* ./build/lib &&\
    sudo cp -R ./build/ /usr/local/lib/python3.6/dist-packages/mish_mish-0.0.3-py3.6-linux-x86_64.egg/mish_mish/

# pip install the project
COPY . /workspace/
RUN pip install --user -e .
