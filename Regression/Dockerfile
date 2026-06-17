FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model files
COPY frontend.py .
COPY fmodel.sav .
COPY scaler.sav .
COPY .streamlit/ .streamlit/

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.headless", "true"]
