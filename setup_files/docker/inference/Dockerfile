FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

COPY requirements.txt /var/azureml-app/
RUN pip install -r /var/azureml-app/requirements.txt --no-cache-dir