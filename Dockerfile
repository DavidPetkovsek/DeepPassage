FROM tensorflow/tensorflow:2.0.0-gpu-py3
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python tqdm pillow
CMD bash