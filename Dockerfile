FROM python/python:3.10-bookworm 
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r ./requirements.txt
COPY app.py /app
CMD ["python", "app.py"]