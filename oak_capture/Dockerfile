FROM luxonis/depthai-library:latest

RUN apt-get install -y ffmpeg gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgirepository1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-plugins-good gstreamer1.0-plugins-base

# RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | tee /etc/udev/rules.d/80-movidius.rules
# RUN udevadm control --reload-rules && udevadm trigger


COPY ./requirements.txt .

RUN python3 -m pip install -r requirements.txt

WORKDIR /app

ADD ./oak_d/ /app/oak_d/
ADD ./processingPipelines/ /app/processingPipelines/
ADD ./data/ /app/data/
COPY ./main.py ./setup_logging.py /app/
COPY ./dummy_main.py /app/

CMD [ "python3", "main.py" ]
# CMD [ "python3", "dummy_main.py"]
