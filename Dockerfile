FROM nvidia/cuda:9.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    	    build-essential \
	    pkg-config \
	    gfortran \
	    libatlas-base-dev \
	    fonts-lyx \
	    libfreetype6-dev \
	    libpng-dev \
	    sudo \
	    python3 \
	    python3-pip \
	    python3-setuptools\
	    imagemagick\
	    wget

RUN python3 -m pip install -U pip
RUN python3 -m pip install numpy
RUN python3 -m pip install pandas
RUN python3 -m pip install torch
RUN python3 -m pip install torchvision
RUN python3 -m pip install sklearn
RUN python3 -m pip install matplotlib
RUN python3 -m pip install ipdb

ADD train_models.py /
ADD eval_test.py /
ADD run_all.sh /
ADD src /src

RUN chmod +x run_all.sh

RUN mkdir /models
RUN mkdir /preds
RUN mkdir /logfiles

CMD [ "./run_all.sh" ]
