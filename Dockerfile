FROM python:3.9-buster
COPY ./ /app
WORKDIR /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN curl https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth -o /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
CMD ["python3","googlenet.py"]
