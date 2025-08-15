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


class ChatResponse(BaseModel):
  text: str
  tool_invocations: List[Dict[str, Any]] = []


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
  raise RuntimeError("OPENAI_API_KEY not set")

OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

CAL_API_KEY = os.getenv("CAL_API_KEY") or os.getenv("CALCOM_API_KEY")
if not CAL_API_KEY:
  # We keep the server up to let non-tool conversation work, but tools will raise until configured
  print("[warn] CAL_API_KEY/CALCOM_API_KEY is not set. Cal.com tool calls will fail until configured.")

# Support both CAL_BASE_URL and CALCOM_BASE_URL
CAL_BASE_URL = os.getenv("CAL_BASE_URL") or os.getenv("CALCOM_BASE_URL") or "https://api.cal.com/v1"

# If user provided a booking page URL, convert it to API format
if "cal.com/" in CAL_BASE_URL and not CAL_BASE_URL.startswith("https://api."):
  # Extract username from booking URL like cal.com/alex-rao-byjxbi/
  if CAL_BASE_URL.startswith("cal.com/"):
    CAL_BASE_URL = "https://api.cal.com/v1"
  elif "cal.com/" in CAL_BASE_URL:
    CAL_BASE_URL = "https://api.cal.com/v1"

CALCOM_BASE_URL = CAL_BASE_URL.rstrip("/")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Cal.com Function-Calling Chatbot", version="0.1.0")

@app.on_event("startup")
async def startup_event():
  """Pre-fetch user email on server startup"""
  if CAL_API_KEY:
    await get_user_email()

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

# Global user info cache
USER_EMAIL: Optional[str] = None


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
  if session_id not in SESSION_ID_TO_MESSAGES:
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    system_preamble = (
      f"IMPORTANT: You are {OPENAI_MODEL}, NOT GPT-3. Do not claim to be GPT-3 or any other model. "
      f"You are an advanced AI assistant connected to the Cal.com API. "
      f"Today's date is {current_date} and the current time is {current_time}. "
      "You can list event types, list a user's bookings by email, create bookings, cancel bookings, and reschedule by cancel+rebook. "
      "ALWAYS start by listing available event types when a user first asks about booking, so they know what options are available. "
      "ONLY suggest event types that actually exist in the user's Cal.com account - never suggest generic types like 'webinar' or 'consultation' unless they appear in the event types list. "
      "When details are missing (e.g., date/time, event type), ask follow-up questions. Email is automatically retrieved from the account - do not ask users for their email. "
      "Use today's date as reference for relative time expressions like 'tomorrow', 'next week', etc. "
      "When creating bookings, be very careful with timezones - always confirm the user's timezone and use proper ISO 8601 format with timezone offset (e.g., '2024-01-15T23:00:00-08:00' for 11 PM PST). "
      "Remember that PST is UTC-8 and PDT is UTC-7. Never convert times to UTC unless specifically requested. "
      "If you encounter 'no_available_users_found_error', explain that the user needs to set up their Cal.com account properly first (user profile, availability, event types). "
      "If you encounter 'availability_conflict' errors, clearly explain that the requested time is outside the user's configured availability hours and suggest alternative times within their availability window. "
      "If you encounter 'scheduling_conflict' errors, explain that the time slot is already booked and use the get_available_slots tool to suggest alternative times near the requested time. "
      "If asked about your model, state that you are accessing the OpenAI API but cannot reliably self-identify your exact model version."
    )
    SESSION_ID_TO_MESSAGES[session_id] = [
      {"role": "system", "content": system_preamble},
    ]
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
  # GET /event-types - this should list all event types available to your API key
  return await cal_request("GET", "/event-types")


async def tool_list_users() -> Dict[str, Any]:
  # GET /users - list all users in your Cal.com account
  return await cal_request("GET", "/users")


async def tool_get_available_slots(event_type_id: int, start_date: str, end_date: str, timezone: str = "America/Los_Angeles") -> Dict[str, Any]:
  # GET /slots - get available time slots for an event type
  params = {
    "eventTypeId": event_type_id,
    "startTime": start_date,
    "endTime": end_date,
    "timeZone": timezone
  }
  return await cal_request("GET", "/slots", params=params)


async def get_user_email() -> Optional[str]:
  """Fetch user email from Cal.com API and cache it globally"""
  global USER_EMAIL
  if USER_EMAIL is not None:
    return USER_EMAIL
    
  try:
    users_data = await cal_request("GET", "/users")
    print(f"[DEBUG] Users API response: {users_data}")
    users = users_data.get("users", []) if isinstance(users_data, dict) else []
    if users and len(users) > 0:
      USER_EMAIL = users[0].get("email")
      print(f"[INFO] Retrieved user email: {USER_EMAIL}")
      print(f"[DEBUG] First user data: {users[0]}")
      return USER_EMAIL
  except Exception as e:
    print(f"[WARN] Could not retrieve user email: {e}")
  
  return None


async def tool_list_bookings_by_email(email: str) -> Dict[str, Any]:
  # GET /bookings?email=...
  params = {"email": email}
  return await cal_request("GET", "/bookings", params=params)


async def tool_create_booking(**kwargs: Any) -> Dict[str, Any]:
  # POST /bookings
  # Expected minimal fields: eventTypeId, start, end, name (email is optional)
  # Cal.com API expects specific field names and formats
  
  # Get user email for booking
  email = kwargs.get("email")
  if not email:
    email = await get_user_email()
  
  # Build the core payload structure that Cal.com v1 expects
  # Use the modern 'responses' format (not legacy top-level name/email)
  payload = {
    "eventTypeId": kwargs.get("eventTypeId"),
    "start": kwargs.get("start"),
    "end": kwargs.get("end"),
    "responses": {
      "name": kwargs.get("name"),
    }
  }
  
  # Add email to responses if available
  if email:
    payload["responses"]["email"] = email
  
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
  
  # Try adding some fields that might be required for user availability
  # Add user/host information if we can get it
  try:
    users_data = await cal_request("GET", "/users")
    print(f"[DEBUG] Available users: {users_data}")
    
    if isinstance(users_data, dict) and "users" in users_data:
      users = users_data["users"]
      if users and len(users) > 0:
        first_user = users[0]
        # Try adding user ID or other identifying info
        if "id" in first_user:
          payload["userId"] = first_user["id"]
        print(f"[DEBUG] Added userId {first_user.get('id')} to payload")
  except Exception as e:
    print(f"[DEBUG] Could not fetch users for debugging: {e}")
  
  # Debug logging
  print(f"[DEBUG] Final booking payload: {payload}")
    
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
      "name": "list_users",
      "description": "List all users in the Cal.com account to see who can host bookings",
      "parameters": {
        "type": "object",
        "properties": {},
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "get_available_slots",
      "description": "Get available time slots for an event type within a date range to suggest alternative times",
      "parameters": {
        "type": "object",
        "properties": {
          "event_type_id": {"type": "integer", "description": "Event type ID to check availability for"},
          "start_date": {"type": "string", "description": "Start date for checking availability (ISO8601 format)"},
          "end_date": {"type": "string", "description": "End date for checking availability (ISO8601 format)"},
          "timezone": {"type": "string", "description": "Timezone like 'America/Los_Angeles'", "default": "America/Los_Angeles"},
        },
        "required": ["event_type_id", "start_date", "end_date"],
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
        "Create a new booking. Requires eventTypeId, start and end ISO8601 timestamps, and attendee name. Email is automatically retrieved from the account."
      ),
      "parameters": {
        "type": "object",
        "properties": {
          "eventTypeId": {"type": "integer"},
          "start": {"type": "string", "description": "ISO8601 start"},
          "end": {"type": "string", "description": "ISO8601 end"},
          "name": {"type": "string", "description": "Attendee name"},
          "title": {"type": "string"},
          "timeZone": {"type": "string", "description": "Timezone like 'America/New_York'"},
          "language": {"type": "string", "description": "Language code like 'en'"},
          "notes": {"type": "string"},
        },
        "required": ["eventTypeId", "start", "end", "name"],
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
  if name == "list_users":
    return await tool_list_users()
  if name == "get_available_slots":
    return await tool_get_available_slots(
      event_type_id=args.get("event_type_id"),
      start_date=args.get("start_date"),
      end_date=args.get("end_date"),
      timezone=args.get("timezone", "America/Los_Angeles")
    )
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


@app.post("/clear-session")
async def clear_session(session_id: str = "web") -> Dict[str, Any]:
  """Clear conversation history for a session"""
  if session_id in SESSION_ID_TO_MESSAGES:
    del SESSION_ID_TO_MESSAGES[session_id]
  return {"cleared": True, "session_id": session_id}


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
  messages = get_session_messages(req.session_id)
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
        error_msg = str(err)
        
        # Provide helpful guidance for common Cal.com errors
        if "no_available_users_found_error" in error_msg:
          # Check if this is a booking error and if we can actually list users
          # This suggests the account setup is fine, but there's an API issue
          if tool_name == "create_booking":
            try:
              users_check = await cal_request("GET", "/users")
              if users_check and isinstance(users_check, dict) and users_check.get("users"):
                error_result = {
                  "error": "no_available_users_found_error",
                  "tool_name": tool_name,
                  "guidance": "The Cal.com account setup appears correct (users found), but booking creation is failing. This may be a temporary API issue or the event type may not be properly configured. Try again or check if the event type is assigned to a user.",
                  "debug_info": f"Users found: {len(users_check.get('users', []))}"
                }
              else:
                error_result = {
                  "error": "no_available_users_found_error", 
                  "tool_name": tool_name,
                  "guidance": "You need to set up a user/host in your Cal.com account first. Go to your Cal.com dashboard and ensure you have: 1) A user profile created, 2) Availability windows configured, 3) Event types properly assigned to a user."
                }
            except Exception:
              error_result = {
                "error": "no_available_users_found_error", 
                "tool_name": tool_name,
                "guidance": "You need to set up a user/host in your Cal.com account first. Go to your Cal.com dashboard and ensure you have: 1) A user profile created, 2) Availability windows configured, 3) Event types properly assigned to a user."
              }
          else:
            error_result = {
              "error": "no_available_users_found_error", 
              "tool_name": tool_name,
              "guidance": "You need to set up a user/host in your Cal.com account first. Go to your Cal.com dashboard and ensure you have: 1) A user profile created, 2) Availability windows configured, 3) Event types properly assigned to a user."
            }
        elif "event_type_not_found" in error_msg:
          error_result = {
            "error": "event_type_not_found",
            "tool_name": tool_name, 
            "guidance": "The event type doesn't exist. First list available event types to see what's available."
          }
        elif any(phrase in error_msg.lower() for phrase in ["already booked", "time slot", "conflict", "overlapping", "not available at this time"]):
          error_result = {
            "error": "scheduling_conflict",
            "tool_name": tool_name,
            "original_error": error_msg,
            "guidance": "This time slot is already booked or conflicts with an existing meeting. Please choose a different time or check your calendar for available slots."
          }
        elif any(phrase in error_msg.lower() for phrase in ["not available", "outside availability", "availability", "business hours", "invalid time", "outside working hours"]):
          error_result = {
            "error": "availability_conflict",
            "tool_name": tool_name,
            "original_error": error_msg,
            "guidance": "The requested time is outside your configured availability hours. Please choose a time within your available booking windows or update your availability settings."
          }
        else:
          error_result = {"error": error_msg, "tool_name": tool_name}
          
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

