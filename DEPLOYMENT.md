# DiagramIQ — Production Deployment Guide 🚀

DiagramIQ is a Neuro-Symbolic Multimodal Reasoning System designed for software architecture, UML, and design diagrams. This guide provides step-by-step instructions for deploying DiagramIQ across local environments, Docker containers, and cloud platforms.

---

## 🏗️ Deployment Architecture

DiagramIQ packages the complete full-stack experience into a unified, high-performance service:
- **Frontend**: React application compiled into optimized static assets.
- **Backend & WSGI**: Flask + Waitress/Gunicorn serving the 6-stage neuro-symbolic reasoning pipeline and static assets from a single port (`5000`).
- **Vision & Reasoning**: EasyOCR + PyTorch + NetworkX + Ollama VLM integration (`llava:latest`).

---

## ⚡ Option 1: 1-Click Local Deployment (Recommended)

### On Windows:
Double-click `deploy_local.bat` or run in PowerShell/CMD:
```bat
deploy_local.bat
```

### On Linux / macOS:
```bash
chmod +x deploy_local.sh
./deploy_local.sh
```

**What it does:**
1. Verifies and installs Python requirements (`waitress`, `easyocr`, `networkx`, `flask-cors`, etc.).
2. Builds the React frontend into `frontend/build/` if not already present.
3. Launches the multi-threaded Waitress WSGI production server at **`http://localhost:5000`**.

---

## 🐳 Option 2: Docker Deployment

### 1. Build and Run Standalone Container
```bash
# Build Docker image
docker build -t diagramiq:latest .

# Run container on port 5000
docker run -d -p 5000:5000 --name diagramiq-app diagramiq:latest
```

Open `http://localhost:5000` in your browser.

---

## 📦 Option 3: Full Stack with Docker Compose (DiagramIQ + Ollama)

To run DiagramIQ alongside a dedicated Ollama container:

```bash
docker compose up -d
```

### Pull Vision Model in Ollama:
```bash
docker exec -it diagramiq-ollama ollama pull llava:latest
```

### Stop Services:
```bash
docker compose down
```

---

## ☁️ Option 4: Cloud Platform Deployment

### A. Deploy to Render.com (via GitHub)
1. Push your repository to GitHub.
2. Log into [Render.com](https://render.com) and click **New +** -> **Blueprint**.
3. Connect your repository — Render will automatically read `render.yaml` and `Dockerfile`.
4. Click **Apply**. Your app will be live at `https://<your-app>.onrender.com`.

### B. Deploy to Railway.app
1. Go to [Railway.app](https://railway.app) and create a **New Project**.
2. Select **Deploy from GitHub repo**.
3. Railway automatically detects `Dockerfile` and `Procfile`.
4. Add environment variable:
   - `PORT=5000`
5. Railway provisions a live public HTTPS endpoint.

### C. Deploy to GCP Cloud Run / AWS ECS
```bash
# Tag for Google Artifact Registry or AWS ECR
docker tag diagramiq:latest gcr.io/<YOUR-PROJECT-ID>/diagramiq:latest
docker push gcr.io/<YOUR-PROJECT-ID>/diagramiq:latest

# Deploy to Cloud Run
gcloud run deploy diagramiq \
  --image gcr.io/<YOUR-PROJECT-ID>/diagramiq:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000 \
  --memory 2Gi \
  --cpu 2
```

---

## 🛠️ Option 5: Development Mode (Hot Reloading)

If you wish to make live code modifications during development:

1. **Terminal 1 (Backend)**:
   ```bash
   python backend/app.py
   ```
2. **Terminal 2 (Frontend)**:
   ```bash
   cd frontend
   npm start
   ```
   Access React hot-reload dev server at `http://localhost:3000`.

---

## ⚙️ Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `5000` | HTTP port the server binds to |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for Ollama VLM API |
| `OLLAMA_TAGS_URL` | `http://localhost:11434/api/tags` | Ollama model tags discovery endpoint |
| `REACT_APP_API_BASE`| `""` (in prod) / `http://localhost:5000` (in dev) | Backend API base URL for the frontend |

---

## 🔍 Verification & Health Checks

Once deployed, verify health by querying:
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "DiagramIQ Multimodal Reasoning System",
  "version": "1.0",
  "ollama_connected": true,
  "supported_diagrams": [
    "UML_CLASS", "UML_SEQUENCE", "SW_ARCHITECTURE", "FLOWCHART", "ER_DIAGRAM", "NETWORK_TOPOLOGY"
  ]
}
```
