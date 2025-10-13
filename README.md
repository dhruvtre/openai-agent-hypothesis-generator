# Hypothesis Generator

AI research assistant that generates testable hypotheses using OpenAI Agent SDK with multi-provider support, literature search, and a modern web interface.

## Quick Start

### Backend
1. `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. Run: `python app.py`

### Frontend
1. `cd frontend`
2. Install dependencies: `npm install`
3. Run: `npm run dev`
4. Open: `http://localhost:3000/chat`

## Features

- **Multi-Provider**: OpenAI and OpenRouter (Claude, Llama, etc.)
- **Streaming Output**: Real-time hypothesis generation
- **Literature Search**: Academic research integration  
- **Web Interface**: Modern React UI with tool call visibility
- **Structured JSON**: Citations and compute budgets included

## Configuration

Set in `backend/.env`:
- `MODEL_PROVIDER`: `"openai"` or `"openrouter"`
- `OPENAI_API_KEY`: Required for OpenAI models
- `OPENROUTER_API_KEY`: Required for OpenRouter models

## Structure

- `backend/` - Python FastAPI server and agent logic
- `frontend/` - Next.js React web interface

## Output Format

Each hypothesis includes:
- Falsifiable claim
- Specific dataset and baseline
- Success metrics and thresholds
- Compute budget (≤6 GPU hours)
- Academic citations