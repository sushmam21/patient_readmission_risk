FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime pieces
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

EXPOSE 8000
WORKDIR /app/src
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
