FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY loitering_detector.py .

# Create uploads directory with proper permissions
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

# Pre-download YOLO model during build (avoids runtime download delays)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && mv /root/.config /home/appuser/.config 2>/dev/null || true \
    && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:7860", "--timeout", "120", "server:app"]
