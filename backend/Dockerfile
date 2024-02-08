FROM nvidia/cudagl:11.4.2-base

WORKDIR /ai

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
        ffmpeg \
        unzip \
        python \
        python3-pip \
        python3-venv \
        libopencv-core-dev \
        libopencv-highgui-dev \
        libopencv-imgproc-dev \
        libopencv-video-dev \
        mesa-common-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev \
        software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && apt-get install -y openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate

RUN pip3 install --upgrade setuptools
RUN pip3 install future

# Install bazel
ARG BAZEL_VERSION=5.2.0
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/b\
azel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

RUN pip3 install --upgrade pip
RUN pip3 install scikit-build
RUN pip3 install cmake

RUN pip3 install numpy
RUN pip3 install opencv-python

COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY . /ai/
EXPOSE 8000
RUN touch WORKSPACE

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install python-multipart
COPY . /code/app

WORKDIR /code/app

# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8689"]

