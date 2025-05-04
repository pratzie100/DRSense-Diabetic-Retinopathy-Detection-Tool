FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r backend-requirements.txt
ENV PORT=8080
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]