
FROM python:3.12.13-slim

#Ensure that Python output is sent straight to the terminal without buffering
ENV PYTHONBUFFERED=1 
#Ensure that Python creates .pyc files in the same directory as the source .py files
ENV PYTHONBYTECODE=1 


WORKDIR /app


COPY requirements_script_training.txt .


RUN pip install -r requirements_script_training.txt

COPY scripts/*.py ./
COPY scripts/config.yaml ./
COPY data/heart.csv data/.


CMD ["bash", "-c", "python train.py && python register_model.py"]
