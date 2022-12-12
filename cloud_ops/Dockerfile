FROM python:3.9.16-alpine

COPY ./requirements.txt .

RUN python3 -m pip install -r requirements.txt

WORKDIR /app

COPY ./main.py /app/main.py

CMD [ "python3", "main.py" ]
