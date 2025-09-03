
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

ENTRYPOINT ["functions-framework", "--source=main.py", "--target=main", "--port=8080"]
