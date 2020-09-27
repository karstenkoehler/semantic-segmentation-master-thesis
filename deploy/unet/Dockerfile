FROM tensorflow/tensorflow:2.2.0-gpu

WORKDIR /thesis
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && apt-get install -y python-opencv
RUN pip install opencv-python

ADD models/unet-23D.hdf5 /thesis/model/model.hdf5
ADD app.py /thesis/code/app.py

CMD ["python", "code/app.py"]