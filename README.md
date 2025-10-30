# Proposal Generator API

A FastAPI backend to store sample proposals and generate new job proposals using OpenAI and Claude AI.

## Usage

1. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API:
   
   ```bash
   uvicorn main:app --reload
   ```
3. Use the endpoints to upload sample proposals and generate new ones.

---

## Frontend
- Served at `/app` when the server is running.
- Open `http://127.0.0.1:8000/app` in your browser.

---

## Environment Variables
Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
CLAUDE_API_KEY=your-claude-key
# Optional model overrides
# OPENAI_MODEL=gpt-4o-mini
# CLAUDE_MODEL=claude-3-sonnet-20240229
```

---

## Deploy with Docker (Local)

1) Build the image
```bash
docker build -t proposal-gen:latest .
```

2) Run the container (pass your env file)
```bash
docker run -p 8000:8000 --env-file .env -v %cd%/proposal_store.json:/app/proposal_store.json proposal-gen:latest
```
Windows PowerShell: replace `%cd%` with your full path, e.g.
```bash
docker run -p 8000:8000 --env-file .env -v "D:/Practice/Make new Proposal/proposal_store.json:/app/proposal_store.json" proposal-gen:latest
```

3) Open in browser
```
http://127.0.0.1:8000/app
```

Note: The bind mount persists your `proposal_store.json` locally.

---

## Deploy to a PaaS (Render/Railway/Fly.io)

General steps (Render example):
- Create a new Web Service from this repo.
- Runtime: Docker.
- Expose port: 8000
- Environment variables: add `OPENAI_API_KEY` and `CLAUDE_API_KEY`.
- Persistent file: `proposal_store.json` is file-based. Prefer a persistent disk or migrate to a DB for production.

If the platform requires a start command, use:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Health Check
- API: `GET /proposals` should return JSON array.
- Frontend: `GET /app` should serve the UI.

---

Further endpoints (for proposal generation) to be added after basic setup.
