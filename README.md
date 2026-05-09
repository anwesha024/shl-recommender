# SHL Assessment Recommender

Conversational FastAPI agent for recommending SHL Individual Test Solutions.

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → conversational reply + optional recommendations

## Quick Start (local)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deploy on Render

1. Push this folder to a GitHub repo
2. Go to render.com → New Web Service → connect repo
3. Set `ANTHROPIC_API_KEY` as environment secret
4. Deploy — Render uses `render.yaml` automatically

## Example Request

```bash
curl -X POST https://your-service.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hiring a mid-level Java developer who works with stakeholders"}]}'
```

## Project Structure

- `main.py` — FastAPI app, system prompt, validation layer
- `catalog.json` — SHL Individual Test Solutions (59 assessments)
- `test_agent.py` — Behavioural test suite (requires running server)
- `render.yaml` — Render deployment config
- `Dockerfile` — Container deployment alternative

## Catalog Coverage

59 Individual Test Solutions covering:
- **Ability (A)**: Verify Numerical/Verbal/Inductive/Deductive/Mechanical/Spatial, Throughput variants
- **Personality (P)**: OPQ32r, OPQ32, MQ, CCSQ, Work Strengths, DSI, UCR, SPQ
- **Knowledge (K)**: Java, Python, SQL, .NET, JavaScript, React, Angular, C++, C#, AWS, Azure, ML, Data Analysis, Cybersecurity, Excel, Agile, Accounting
- **Situational Judgement (S)**: Supervisory, Customer Service, Contact Center, Manager, Graduate Scenarios
- **Biodata/Language (B)**: SVAR Spoken English, Workplace English Test
- **Combined batteries**: Automata Pro, Administrative Professional, Technology Professional, Entry Level Sales
