FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# Mount your training repo read-only at /training (config.py, models.py, checkpoints/)
ENV UNO_TRAINING_DIR=/training
EXPOSE 8000
# workers=1 — sessions live in-process
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]