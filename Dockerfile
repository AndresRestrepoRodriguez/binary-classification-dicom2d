FROM python:3.9-slim

docker images


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api.py"]