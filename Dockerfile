ARG CUDA_VERSION=10.0
ARG CENTOS_VERSION=7
FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-centos${CENTOS_VERSION}

RUN yum -y install \
    libcurl4-openssl-dev \
    wget \
    zlib-devel \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    make

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

RUN cd /opt && git clone --recursive https://github.com/opencv/opencv.git &&  cd /opt/opencv && git checkout 3.4 && mkdir build && cd build && cmake .. && make -j 10 && make install

RUN yum install -y libSM libXrender libXext && pip install opencv-python mxnet-cu100

ADD TensorRT-7.0.0.11.CentOS-7.6.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz /opt/

RUN cd /opt && export TRT_RELEASE=`pwd`/TensorRT-7.0.0.11 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_RELEASE/lib &&  git clone -b master https://github.com/nvidia/TensorRT TensorRT -b release/7.0 && cd TensorRT && git submodule update --init --recursive && export TRT_SOURCE=`pwd` && cd $TRT_SOURCE && mkdir -p build && cd build && cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=10.0 && make -j$(nproc) && make install

RUN pip install /opt/TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl && pip install onnx==1.6.0

RUN pip install opt/TensorRT-7.0.0.11/uff/uff-0.6.5-py2.py3-none-any.whl pytest==5.1.2 && cd /opt/ && git clone --recursive https://github.com/onnx/onnx-tensorrt.git
RUN cd /opt && git clone -b 3.8.x --recursive https://github.com/protocolbuffers/protobuf.git protobuf3.8.x
RUN yum install autoconf automake libtool swig3.x86_64 python3-devel -y
RUN cd /opt/protobuf3.8.x && ./autogen.sh &&./configure && make -j 10 && make install && ldconfig
RUN cd /opt/onnx-tensorrt && mkdir build && cd build && cmake -DTENSORRT_ROOT=/opt/TensorRT-7.0.0.11  .. && make -j 10 && make install && ldconfig && cd .. && export TRT_ROOT=/opt/TensorRT-7.0.0.11/ && sed -i '/#ifndef NV_ONNX_PARSER_H/i #define TENSORRTAPI' NvOnnxParser.h && python setup.py build && python setup.py install
```

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-7.0.0.11/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-7.0.0.11/lib/