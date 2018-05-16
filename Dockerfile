
FROM python:3.6

ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        git \
        sudo \
        wget \
        curl \
        vim \
        bzip2 \
        xvfb \
        libx11-6 \
        ca-certificates \
        libopenblas-base \
        graphviz \
        && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip

RUN pip3 --no-cache-dir install --upgrade \
        scipy \
        seaborn \
        pandas \
        h5py \
        graphviz \
        pydot \
        pyaml \
        nose \
        coverage \
        nose-watch \
        tqdm==4.19.5 \
        torch==0.4.0 \
        matplotlib==2.0.2 \
        tensorflow==1.5.0 \
        keras==2.1.4 \
        numpy==1.13.3 \
        torchvision==0.2.1 \
        jupyter

# Expose the JupterNotebook port
EXPOSE 8888

RUN mkdir /app
COPY . /app

WORKDIR /app

CMD xvfb-run -s "-screen 0 640x480x24" jupyter notebook --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token=''
