# just following the basic pattern for running a fastapi app in docker

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# building the chromadb index now (at build time) so the container is
# ready to go as soon as it starts, instead of doing it on first request
RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
