FROM luxonis/depthai-library:latest

COPY ./requirements.txt .

RUN python3 -m pip install -r requirements.txt

WORKDIR /app

ADD ./oak_d/ /app/oak_d/
ADD ./processingPipelines/ /app/processingPipelines/
ADD ./data/ /app/data/
COPY ./displayPipeline.py ./main.py /app/

# CMD [ "python3", "main.py" ]