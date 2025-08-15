# Cal.com Function-Calling Chatbot (FastAPI)

This is a minimal FastAPI backend that uses OpenAI tool/function calling to integrate with the Cal.com v1 API.

## Features
- Understands natural language requests to:
  - List users
  - Book meetings
    - Needs timezone, time, date, and type (15 min, 30 min, etc.)
  - List meetings
  - Cancel Events

## Setup
1. Create a Python 3.10+ virtual environment, `python -m venv venv`
2. Install deps:
```bash
pip install -r requirements.txt
```
3. Create `.env` (root or `pychatbot/`) with:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CAL_API_KEY=cal_test_...  # or cal_live_...
CAL_BASE_URL=cal.com/...
```

## Run
```bash
uvicorn app:app --reload --port 5050
```

## API
- `GET /health` – returns basic config
- `POST /chat`
```json
{
  "session_id": "demo",
  "message": "help me book a 30 min meeting tomorrow morning",
  "user_email": "alice@example.com"
}
```

## Notes
- Tool schemas are defined in code and mapped to Cal.com endpoints per their v1 docs: [Cal.com API v1](https://cal.com/docs/api-reference/v1/introduction)
- For production, replace in-memory session storage and add authentication.

