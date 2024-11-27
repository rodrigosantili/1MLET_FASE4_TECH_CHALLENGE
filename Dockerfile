FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY res/ /app/res/
EXPOSE 8080
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
