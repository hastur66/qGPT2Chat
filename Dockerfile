FROM python:3.10-bookworm 
# FROM tensorflow/tensorflow:2.12.0
WORKDIR /app
COPY app.py requirements.txt /app
RUN pip install -r ./requirements.txt
EXPOSE 7860
CMD ["python3", "app.py"]
