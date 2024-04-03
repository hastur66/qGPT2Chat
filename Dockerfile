FROM python:3.10-bookworm 
# FROM tensorflow/tensorflow:2.12.0
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r ./requirements.txt
COPY app.py /app
EXPOSE 8000
CMD ["python3", "app.py"]
