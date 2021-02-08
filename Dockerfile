FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN apt-get update && apt-get install -y \ 
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \  
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \ 
    libgtk2.0-dev \ 
    python-numpy \ 
    python-pycurl \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    python-opencv \ 
    qt5-default \
    libvtk6-dev \ 
    tesseract-ocr \
    zlib1g-dev \
    ffmpeg
COPY ./app /app/app
COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt
