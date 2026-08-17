# ── Stage 1: Build React Frontend ───────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit

COPY frontend/ ./
ENV NODE_ENV=production
RUN npm run build

# ── Stage 2: Production Python & Neuro-Symbolic Engine ────────────────────────
FROM python:3.11-slim

# System dependencies for OpenCV, EasyOCR, and Image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code & sample data
COPY backend/ ./backend/
COPY samples/ ./samples/
COPY requirements.md .

# Copy prebuilt frontend assets into the image
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Create data directories
RUN mkdir -p /app/data/diagrams

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV OLLAMA_BASE_URL="http://ollama:11434"

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Launch with Gunicorn production WSGI server
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "backend.wsgi:app"]
