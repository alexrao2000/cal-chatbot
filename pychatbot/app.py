import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class ChatRequest(BaseModel):
  session_id: str = Field(..., description="Client-chosen session identifier to keep conversation state")
  message: str = Field(..., description="User input text")
  user_email: Optional[str] = Field(None, description="Optional user email; helps with listing/canceling bookings")


class ChatResponse(BaseModel):
  text: str
  tool_invocations: List[Dict[str, Any]] = []


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
  raise RuntimeError("OPENAI_API_KEY not set")

OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

CAL_API_KEY = os.getenv("CAL_API_KEY")
if not CAL_API_KEY:
  # We keep the server up to let non-tool conversation work, but tools will raise until configured
  print("[warn] CAL_API_KEY is not set. Cal.com tool calls will fail until configured.")

CALCOM_BASE_URL = (os.getenv("CALCOM_BASE_URL") or "https://api.cal.com/v1").rstrip("/")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Cal.com Function-Calling Chatbot", version="0.1.0")

# CORS for local dev (Vite, etc.)
app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)


# In-memory conversation state. For production, use a store like Redis keyed by session_id.
SESSION_ID_TO_MESSAGES: Dict[str, List[Dict[str, Any]]] = {}


def get_session_messages(session_id: str, user_email: Optional[str]) -> List[Dict[str, Any]]:
  if session_id not in SESSION_ID_TO_MESSAGES:
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    system_preamble = (
      f"You are GPT-4o, an advanced AI assistant connected to the Cal.com API. "
      f"Today's date is {current_date} and the current time is {current_time}. "
      f"You are running the {OPENAI_MODEL} model. "
      "You can list event types, list a user's bookings by email, create bookings, cancel bookings, and reschedule by cancel+rebook. "
      "When details are missing (e.g., email, date/time, event type), ask follow-up questions. "
      "Use today's date as reference for relative time expressions like 'tomorrow', 'next week', etc. "
      "Prefer ISO 8601 times and always include timezone information when creating bookings."
    )
    SESSION_ID_TO_MESSAGES[session_id] = [
      {"role": "system", "content": system_preamble},
    ]
    if user_email:
      SESSION_ID_TO_MESSAGES[session_id].append({
        "role": "system",
        "content": f"User email (if needed for bookings): {user_email}",
      })
  return SESSION_ID_TO_MESSAGES[session_id]


async def cal_request(method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                      json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  if not CAL_API_KEY:
    raise HTTPException(status_code=500, detail="CAL_API_KEY not configured")
  url = f"{CALCOM_BASE_URL}{path}"
  # Cal.com v1 docs show apiKey via query param.
  # Ref: https://cal.com/docs/api-reference/v1/introduction
  params = dict(params or {})
  params.setdefault("apiKey", CAL_API_KEY)
  headers = {}
  async with httpx.AsyncClient(timeout=30) as http:
    resp = await http.request(method, url, params=params, headers=headers, json=json_body)
    try:
      data = resp.json()
    except Exception:
      data = {"raw": await resp.aread()}
    if resp.status_code >= 400:
      raise HTTPException(status_code=resp.status_code, detail=data)
    return data


# ---- Tool implementations (actual effects) ----

async def tool_list_event_types() -> Dict[str, Any]:
  # GET /event-types
  return await cal_request("GET", "/event-types")


async def tool_list_bookings_by_email(email: str) -> Dict[str, Any]:
  # GET /bookings?email=...
  params = {"email": email}
  return await cal_request("GET", "/bookings", params=params)


async def tool_create_booking(**kwargs: Any) -> Dict[str, Any]:
  # POST /bookings
  # Expected minimal fields: eventTypeId, start, end, name, email
  # Cal.com API expects specific field names and formats
  payload = {
    "eventTypeId": kwargs.get("eventTypeId"),
    "start": kwargs.get("start"),
    "end": kwargs.get("end"),
    "responses": {
      "name": kwargs.get("name"),
      "email": kwargs.get("email"),
    }
  }
  
  # Add optional fields with correct Cal.com naming
  if kwargs.get("timeZone") or kwargs.get("timezone"):
    payload["timeZone"] = kwargs.get("timeZone") or kwargs.get("timezone")
  
  if kwargs.get("language"):
    payload["language"] = kwargs.get("language")
  else:
    payload["language"] = "en"  # Default language
    
  if kwargs.get("metadata"):
    payload["metadata"] = kwargs.get("metadata")
  else:
    payload["metadata"] = {}  # Default empty metadata
    
  if kwargs.get("title"):
    payload["title"] = kwargs.get("title")
    
  if kwargs.get("notes"):
    payload["responses"]["notes"] = kwargs.get("notes")
    
  return await cal_request("POST", "/bookings", json_body=payload)


async def tool_cancel_booking(booking_id: int) -> Dict[str, Any]:
  # DELETE /bookings/{id}
  return await cal_request("DELETE", f"/bookings/{booking_id}")


async def tool_reschedule_booking(booking_id: int, new_start: str, new_end: str) -> Dict[str, Any]:
  # Simple strategy: cancel + recreate is a fallback if dedicated reschedule endpoint is unavailable.
  # 1) GET booking to capture details
  booking = await cal_request("GET", f"/bookings/{booking_id}")
  # 2) Cancel
  await cal_request("DELETE", f"/bookings/{booking_id}")
  # 3) Recreate with adjusted times using original fields when possible
  orig = booking.get("booking") or booking
  payload = {
    "eventTypeId": orig.get("eventTypeId"),
    "start": new_start,
    "end": new_end,
    "name": orig.get("name"),
    "email": orig.get("email"),
    "title": orig.get("title"),
    "timezone": orig.get("timezone"),
    "location": orig.get("location"),
    "notes": orig.get("notes"),
    "userFields": orig.get("userFields"),
  }
  return await cal_request("POST", "/bookings", json_body=payload)


# ---- OpenAI tool schema ----

TOOLS: List[Dict[str, Any]] = [
  {
    "type": "function",
    "function": {
      "name": "list_event_types",
      "description": "List all event types for the authenticated Cal.com account",
      "parameters": {
        "type": "object",
        "properties": {},
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "list_bookings_by_email",
      "description": "List bookings for a given user email",
      "parameters": {
        "type": "object",
        "properties": {
          "email": {"type": "string", "description": "User email address"},
        },
        "required": ["email"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "create_booking",
      "description": (
        "Create a new booking. Requires eventTypeId, start and end ISO8601 timestamps, and attendee info."
      ),
      "parameters": {
        "type": "object",
        "properties": {
          "eventTypeId": {"type": "integer"},
          "start": {"type": "string", "description": "ISO8601 start"},
          "end": {"type": "string", "description": "ISO8601 end"},
          "name": {"type": "string"},
          "email": {"type": "string"},
          "title": {"type": "string"},
          "timeZone": {"type": "string", "description": "Timezone like 'America/New_York'"},
          "language": {"type": "string", "description": "Language code like 'en'"},
          "notes": {"type": "string"},
        },
        "required": ["eventTypeId", "start", "end", "name", "email"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "cancel_booking",
      "description": "Cancel a booking by its numeric id",
      "parameters": {
        "type": "object",
        "properties": {
          "booking_id": {"type": "integer"},
        },
        "required": ["booking_id"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "reschedule_booking",
      "description": (
        "Reschedule a booking by id with new start and end times (ISO8601). Fallbacks to cancel+recreate."
      ),
      "parameters": {
        "type": "object",
        "properties": {
          "booking_id": {"type": "integer"},
          "new_start": {"type": "string"},
          "new_end": {"type": "string"},
        },
        "required": ["booking_id", "new_start", "new_end"],
      },
    },
  },
]


async def dispatch_tool_call(name: str, arguments_json: str) -> Dict[str, Any]:
  try:
    args = json.loads(arguments_json or "{}")
  except Exception:
    args = {}
  if name == "list_event_types":
    return await tool_list_event_types()
  if name == "list_bookings_by_email":
    email = args.get("email")
    return await tool_list_bookings_by_email(email=email)
  if name == "create_booking":
    return await tool_create_booking(**args)
  if name == "cancel_booking":
    return await tool_cancel_booking(booking_id=args.get("booking_id"))
  if name == "reschedule_booking":
    return await tool_reschedule_booking(
      booking_id=args.get("booking_id"),
      new_start=args.get("new_start"),
      new_end=args.get("new_end"),
    )
  raise HTTPException(status_code=400, detail=f"Unknown tool name: {name}")


@app.get("/health")
def health() -> Dict[str, Any]:
  return {"ok": True, "openai_model": OPENAI_MODEL, "calcom_base": CALCOM_BASE_URL, "cal_key": bool(CAL_API_KEY)}


@app.get("/models")
async def list_models() -> Dict[str, Any]:
  """List all models available to your OpenAI API key"""
  if not OPENAI_API_KEY:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
  
  try:
    models_response = client.models.list()
    model_ids = [model.id for model in models_response.data]
    model_ids.sort()
    
    # Filter to just chat models (gpt-* typically)
    chat_models = [m for m in model_ids if 'gpt' in m.lower()]
    
    return {
      "current_model": OPENAI_MODEL,
      "all_models": model_ids,
      "chat_models": chat_models,
      "recommended": [m for m in chat_models if any(x in m for x in ['gpt-4o', 'gpt-4-turbo', 'gpt-4'])]
    }
  except Exception as err:
    raise HTTPException(status_code=500, detail=f"Failed to list models: {str(err)}")


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
  """Legacy Node.js compatibility endpoint"""
  return {"ok": True, "hasKey": bool(OPENAI_API_KEY), "model": OPENAI_MODEL}


class AskRequest(BaseModel):
  prompt: str
  model: Optional[str] = None


@app.post("/api/ask")
async def api_ask(req: AskRequest) -> Dict[str, Any]:
  """Legacy Node.js compatibility endpoint - simple OpenAI call without function calling"""
  if not OPENAI_API_KEY:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")
  
  if not req.prompt or not req.prompt.strip():
    raise HTTPException(status_code=400, detail="prompt (string) is required")
  
  try:
    completion = client.chat.completions.create(
      model=req.model or OPENAI_MODEL,
      messages=[{"role": "user", "content": req.prompt}],
      temperature=0.7,
    )
    text = completion.choices[0].message.content or ""
    return {"text": text}
  except Exception as err:
    # Match Node.js error handling
    status = getattr(err, "status_code", 500)
    message = getattr(err, "message", str(err))
    if hasattr(err, "error") and hasattr(err.error, "message"):
      message = err.error.message
    print(f"OpenAI error: {err}")
    raise HTTPException(status_code=status, detail=message)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
  messages = get_session_messages(req.session_id, req.user_email)
  messages.append({"role": "user", "content": req.message})

  # First model request with tools enabled
  first = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
    temperature=0.3,
  )

  choice = first.choices[0]
  tool_invocations: List[Dict[str, Any]] = []

  if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
    # Include the assistant message that contains tool_calls in the conversation
    assistant_tool_msg: Dict[str, Any] = {
      "role": "assistant",
      "content": choice.message.content,
      "tool_calls": [
        {
          "id": tc.id,
          "type": "function",
          "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in (choice.message.tool_calls or [])
      ],
    }
    messages.append(assistant_tool_msg)
    # For each tool call, execute and provide tool outputs back to the model
    for tool_call in choice.message.tool_calls:
      tool_name = tool_call.function.name
      tool_args_json = tool_call.function.arguments
      try:
        result = await dispatch_tool_call(tool_name, tool_args_json)
        tool_payload = json.dumps(result, ensure_ascii=False)
        tool_invocations.append({"name": tool_name, "args": tool_args_json, "result": result})
      except Exception as err:
        # If tool execution fails, still provide a response to maintain conversation flow
        error_result = {"error": str(err), "tool_name": tool_name}
        tool_payload = json.dumps(error_result, ensure_ascii=False)
        tool_invocations.append({"name": tool_name, "args": tool_args_json, "result": error_result})
        print(f"Tool execution error for {tool_name}: {err}")
      
      # Always add tool response message to maintain proper conversation flow
      messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_payload,
      })

    # Ask the model to synthesize a final user-facing response
    second = client.chat.completions.create(
      model=OPENAI_MODEL,
      messages=messages,
      temperature=0.3,
    )
    final_text = second.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": final_text})
    return ChatResponse(text=final_text, tool_invocations=tool_invocations)

  # No tool calls. Just return the model's text
  final_text = choice.message.content or ""
  messages.append({"role": "assistant", "content": final_text})
  return ChatResponse(text=final_text, tool_invocations=tool_invocations)

