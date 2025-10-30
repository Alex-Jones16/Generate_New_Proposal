from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
import os
from dotenv import load_dotenv
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

app = FastAPI()

# CORS for local/frontend to call backend on Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROPOSAL_STORE = 'proposal_store.json'

def load_proposals():
    if os.path.exists(PROPOSAL_STORE):
        with open(PROPOSAL_STORE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_proposals(proposals):
    with open(PROPOSAL_STORE, 'w', encoding='utf-8') as f:
        json.dump(proposals, f, ensure_ascii=False, indent=2)

async def ai_select_best_proposal(job_desc, proposals):
    """
    Uses OpenAI to choose the most relevant reference proposal based on job description.
    Falls back to first proposal if no key is set or call fails.
    """
    if not OPENAI_API_KEY:
        return proposals[0]
    prompt = (
        "Based on the following job description, choose the most stylistically suited sample proposal.\n"
        "Job Description:\n" + job_desc + "\n" +
        "Sample Proposals:\n" +
        "\n\n".join([f'[{i}] {p["name"]}: {p["content"][:200]}...' for i, p in enumerate(proposals)]) +
        "\n\nProvide ONLY the number of the best sample proposal (in [brackets]):"
    )
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are helpful and concise."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(OPENAI_API_URL, headers=headers, json=body, timeout=40.0)
        if resp.status_code != 200:
            return proposals[0]
        text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "[0]")
        idx = 0
        try:
            idx = int(text.strip().split("[")[-1].split("]")[0])
            if not (0 <= idx < len(proposals)):
                idx = 0
        except:
            idx = 0
        return proposals[idx]
    except Exception:
        return proposals[0]

async def generate_with_openai(job_desc, sample):
    if not OPENAI_API_KEY:
        return "(OpenAI API key not set; returning placeholder text)"
    prompt = (
        f"Here is a sample proposal:\n---\n{sample['content']}\n---\n"
        f"A job description:\n{job_desc}\n"
        "Write a new proposal for this job, closely matching the style and tone of the sample proposal."
    )
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a talented proposal writer."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(OPENAI_API_URL, headers=headers, json=body, timeout=60.0)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = {"error": resp.text}
            return f"(OpenAI error {resp.status_code}) {err}"
        return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "(Empty response)"
    except Exception as e:
        return f"(OpenAI request failed) {e}"

async def generate_with_claude(job_desc, sample):
    if not CLAUDE_API_KEY:
        return "(Claude AI API key not set; returning placeholder text)"
    prompt = (
        f"Here is a sample proposal:\n---\n{sample['content']}\n---\n"
        f"A job description:\n{job_desc}\n"
        "Write a new proposal for this job, closely matching the style and tone of the sample proposal."
    )
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(CLAUDE_API_URL, headers=headers, json=body, timeout=60.0)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = {"error": resp.text}
            return f"(Claude error {resp.status_code}) {err}"
        data = resp.json()
        # Claude v1 messages API returns content as a list of blocks: [{type: 'text', text: '...'}]
        content = data.get("content")
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join([t for t in texts if t]) or "(Empty response)"
        if isinstance(content, str):
            return content
        return "(Empty response)"
    except Exception as e:
        return f"(Claude request failed) {e}"

async def ai_select_top_two(job_desc, proposals):
    """
    Return top two proposals best matching the job_desc.
    Uses OpenAI if available; otherwise falls back to simple keyword overlap.
    """
    if OPENAI_API_KEY:
        prompt = (
            "Based on the following job description, choose the TWO most stylistically suited sample proposals.\n"
            "Return only two indices in ascending order, comma-separated (e.g., 0,3).\n\n"
            "Job Description:\n" + job_desc + "\n\n" +
            "Sample Proposals:\n" +
            "\n\n".join([f'[{i}] {p["name"]}: {p["content"][:200]}...' for i, p in enumerate(proposals)]) +
            "\n\nIndices only:"
        )
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are helpful and concise."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(OPENAI_API_URL, headers=headers, json=body, timeout=45.0)
            if resp.status_code == 200:
                text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "0,1")
                nums = []
                for part in text.replace("[", "").replace("]", "").split(","):
                    try:
                        nums.append(int(part.strip()))
                    except:
                        pass
                nums = [n for n in nums if 0 <= n < len(proposals)]
                nums = sorted(list(dict.fromkeys(nums)))  # unique, keep order then sort
                if len(nums) >= 2:
                    return [proposals[nums[0]], proposals[nums[1]]]
        except Exception:
            pass
    # Fallback: simple keyword overlap
    def tokenize(text):
        return set([w.lower() for w in text.split() if len(w) > 2])
    jd = tokenize(job_desc)
    scores = []
    for i, p in enumerate(proposals):
        s = len(jd & tokenize(p["content"]))
        scores.append((s, i))
    scores.sort(key=lambda x: (-x[0], x[1]))
    a = proposals[scores[0][1]] if scores else proposals[0]
    b = proposals[scores[1][1]] if len(scores) > 1 else proposals[min(1, len(proposals)-1)]
    return [a, b]

@app.post('/proposals')
async def upload_proposal(name: str = Form(...), content: str = Form(...)):
    proposals = load_proposals()
    proposals.append({'name': name, 'content': content})
    save_proposals(proposals)
    return {"message": "Proposal uploaded successfully."}

@app.get('/proposals')
def list_proposals():
    proposals = load_proposals()
    return proposals

@app.post('/generate')
async def generate_proposals(job_description: str = Body(..., embed=True)):
    proposals = load_proposals()
    if not proposals:
        raise HTTPException(status_code=400, detail="No sample proposals available.")
    # pick top two
    top_two = await ai_select_top_two(job_description, proposals)
    sample_a, sample_b = top_two[0], top_two[1]
    # generate four proposals (OpenAI + Claude for each)
    openai_a = await generate_with_openai(job_description, sample_a)
    openai_b = await generate_with_openai(job_description, sample_b)
    claude_a = await generate_with_claude(job_description, sample_a)
    claude_b = await generate_with_claude(job_description, sample_b)
    return {
        "matched_samples": [sample_a, sample_b],
        "results": [
            {"provider": "openai", "sample_index": 0, "text": openai_a},
            {"provider": "openai", "sample_index": 1, "text": openai_b},
            {"provider": "claude", "sample_index": 0, "text": claude_a},
            {"provider": "claude", "sample_index": 1, "text": claude_b}
        ]
    }

@app.get('/favicon.ico')
def favicon():
    return Response(status_code=204)

# Serve frontend at /app
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
