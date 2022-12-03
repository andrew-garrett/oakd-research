FROM luxonis/depthai-library:latest

WORKDIR /app

COPY ../requirements.txt .

RUN python3 pip install -r requirements.txt

ADD ../oak_d/ .
ADD ../processingPipelines .
COPY ../displayPipeline.py main.py .

CMD [ "python3", "main.py" ]