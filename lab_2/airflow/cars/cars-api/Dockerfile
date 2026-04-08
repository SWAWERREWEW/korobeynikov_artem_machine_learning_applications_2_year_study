FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && apt-get install unzip && rm -rf /var/lib/apt/lists/*
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# COPY app.py ml-2023-ratings.zip /
COPY app.py cars.csv /

EXPOSE 8081

ENTRYPOINT ["python"]
CMD ["/app.py"]
