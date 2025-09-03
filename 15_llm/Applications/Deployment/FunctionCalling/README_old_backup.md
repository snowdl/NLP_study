```python
"""
What is Function Calling?
Function calling is a pattern where an LLM interprets natural language into structured intent and parameters, then executes a predefined function (tool) with the correct arguments, and incorporates the result into its reply.
In short: understand the request → decide the action → call a function → answer with the result.

How does it work?

1.User message input
2.Intent detection + argument extraction (e.g., report_card_lost(user_id="U1001"))
3. Function execution (real API/DB/internal system or a mock function)
4.Summarize/post-process the result and present it to the user
"""
```




    '\nWhat is Function Calling?\nFunction calling is a pattern where an LLM interprets natural language into structured intent and parameters, then executes a predefined function (tool) with the correct arguments, and incorporates the result into its reply.\nIn short: understand the request → decide the action → call a function → answer with the result.\n\nHow does it work?\n\n1.User message input\n2.Intent detection + argument extraction (e.g., report_card_lost(user_id="U1001"))\n3. Function execution (real API/DB/internal system or a mock function)\n4.Summarize/post-process the result and present it to the user\n'




```python
#Flow
#Step 1 — Imports & device check
#Step 2 — Mock functions (replace with real APIs later)
   ## What each function does

# `report_card_lost(user_id)`: creates a mock support ticket for a lost/stolen card and returns a JSON payload with a random `ticket_id`.
# `check_card_delivery(card_id)`: returns a mock shipment status (carrier, ETA, last update) for a given card.
#`request_new_card(user_id)`: submits a mock new-card request and returns a JSON payload with a random `order_id`.

#Step 3 — Minimal keyword rules (intent detection)
#Step 4 — Engine (detect → dispatch)
#detect_intent(text, rules) → finds first keyword match and returns
#{"intent","function","args","confidence"} or None.
#case_message(intent_name) → human-readable English case label.
#call_function(intent_spec, **kwargs) → gathers required args (fills safe defaults) and calls the mapped function.
#run_pipeline(text, **kwargs) → end-to-end:
#detect intent
#call function
```


```python
#Environment
```


```python
# Environment & device check
import sys, torch
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device:", device)
```

    device: mps



```python
# ===== Cell 1 — Imports (organized, no duplicates) =====
# Standard library
import sys
import re
import json
from pathlib import Path
from datetime import datetime
import random
```


```python
# ===== Step 2 — Mock functions (replace with real APIs later) =====

def report_card_lost(user_id: str):
    """Mock: create a lost-card ticket."""
    return {
        "action": "report_card_lost",
        "user_id": user_id or "demo_user",
        "ticket_id": f"T{random.randint(100000, 999999)}",
        "status": "received",
        "message": "Your card has been marked as lost. A support agent will contact you shortly."
    }

```


```python
def check_card_delivery(card_id: str):
    """Mock: return a fake shipment status."""
    eta_days = random.choice([1, 2, 3, 4, 5])
    return {
        "action": "check_card_delivery",
        "card_id": card_id or "CARD123",
        "carrier": "MockExpress",
        "eta": f"{eta_days} days",
        "status": "in_transit",
        "last_update": 

```python
def request_new_card(user_id: str):
    """Mock: submit a new card request."""
    return {
        "action": "request_new_card",
        "user_id": user_id or "demo_user",
        "order_id": f"O{random.randint(100000, 999999)}",
        "status": "submitted",
        "message": "A new card request has been submitted.",
    }

```


```python
#Intent Rules (
```


```python
# ===== Step 3 — Minimal keyword rules for intent detection =====
# Each intent maps to:
#   - keywords: phrases to look for in the (lowercased) user text
#   - function: the function to call when this intent matches
#   - args: required argument names for that function
INTENT_RULES = {
    "report_card_lost": {
        "keywords": [
            "lost my card",
            "card stolen",
            "card missing",
            "i lost my card",
            "lost card"
        ],
        "function": "report_card_lost",
        "args": ["user_id"],
    },
    "check_card_delivery": {
        "keywords": [
            "track my card",
            "where is my card",
            "card not arrived",
            "card hasn't arrived",
            "when will my card arrive",
            "card delivery status"
        ],
        "function": "check_card_delivery",
        "args": ["card_id"],
    },
    "request_new_card": {
        "keywords": [
            "order new card",
            "replace my card",
            "new card",
            "request a new card",
            "need a replacement card"
        ],
        "function": "request_new_card",
        "args": ["user_id"],
    },
}

```


```python
# ===== Step 4 — Engine: detect intent & dispatch to a function =====

# Map intent names to actual callable functions (from Step 2)
FUNC_REGISTRY = {
    "report_card_lost": report_card_lost,
    "check_card_delivery": check_card_delivery,
    "request_new_card": request_new_card,
}


```


```python
def detect_intent(text: str, rules: dict):
    """
    Very simple rule-based detector.
    Returns a dict {'intent','function','args','confidence'} or None if no match.
    """
    t = (text or "").lower()
    for name, spec in rules.items():
        for kw in spec.get("keywords", []):
            if re.search(re.escape(kw.lower()), t):
                return {
                    "intent": name,
                    "function": spec["function"],
                    "args": spec.get("args", []),
                    "confidence": 0.9,
                }
    return None

```


```python
# Human-readable English label for each case
CASE_MESSAGE = {
    "report_card_lost":    "Case: Report a lost or stolen card.",
    "check_card_delivery": "Case: Check card delivery status.",
    "request_new_card":    "Case: Request a new or replacement card.",
}
def case_message(intent_name: str) -> str:
    return CASE_MESSAGE.get(intent_name, "Case: Unknown.")

```


```python
def call_function(intent_spec: dict, **kwargs):
    """
    Call the mapped function with required kwargs.
    If an argument is missing, fill a safe default for demo purposes.
    """
    fn = FUNC_REGISTRY[intent_spec["function"]]
    final_kwargs = {}
    for a in intent_spec.get("args", []):
        v = kwargs.get(a)
        if not v:
            v = "demo_user" if a == "user_id" else "CARD123"
        final_kwargs[a] = v
    return fn(**final_kwargs)
```


```python
def run_pipeline(text: str, **kwargs):
    """
    Full pipeline:
      1) detect intent
      2) call function
      3) return a JSON-able dict with an English case message
    """
    intent = detect_intent(text, INTENT_RULES)
    if not intent:
        return {"ok": False, "reason": "no_intent_matched", "text": text}

    result = call_function(intent, **kwargs)
    return {
        "ok": True,
        "input_text": text,
        "detected_intent": intent["intent"],
        "function_called": intent["function"],
        "case_english": case_message(intent["intent"]),
        "result": result,
    }
```


```python
#Quick Sanity Tests
```


```python
# ===== Step 5 — Quick sanity tests =====
# Run a few messages to ensure the pipeline works end-to-end.
# Note: tests are in English to match our keyword rules from Step 3.

tests = [
    {"text": "I lost my card yesterday", "user_id": "U1001"},
    {"text": "Where is my card? It hasn't arrived yet.", "card_id": "C5555"},
    {"text": "Please order a new card", "user_id": "U1001"},
    {"text": "My card was stolen last night", "user_id": "U1002"},
    {"text": "When will my card arrive?", "card_id": "K-9999"},
]

for t in tests:
    out = run_pipeline(**t)
    print(json.dumps(out, ensure_ascii=False, indent=2))
```

    {
      "ok": true,
      "input_text": "I lost my card yesterday",
      "detected_intent": "report_card_lost",
      "function_called": "report_card_lost",
      "case_english": "Case: Report a lost or stolen card.",
      "result": {
        "action": "report_card_lost",
        "user_id": "U1001",
        "ticket_id": "T813567",
        "status": "received",
        "message": "Your card has been marked as lost. A support agent will contact you shortly."
      }
    }
    {
      "ok": true,
      "input_text": "Where is my card? It hasn't arrived yet.",
      "detected_intent": "check_card_delivery",
      "function_called": "check_card_delivery",
      "case_english": "Case: Check card delivery status.",
      "result": {
        "action": "check_card_delivery",
        "card_id": "C5555",
        "carrier": "MockExpress",
        "eta": "2 days",
        "status": "in_transit",
        "last_update": "2025-08-31T08:27:32.242833Z"
      }
    }
    {
      "ok": true,
      "input_text": "Please order a new card",
      "detected_intent": "request_new_card",
      "function_called": "request_new_card",
      "case_english": "Case: Request a new or replacement card.",
      "result": {
        "action": "request_new_card",
        "user_id": "U1001",
        "order_id": "O113849",
        "status": "submitted",
        "message": "A new card request has been submitted."
      }
    }
    {
      "ok": false,
      "reason": "no_intent_matched",
      "text": "My card was stolen last night"
    }
    {
      "ok": true,
      "input_text": "When will my card arrive?",
      "detected_intent": "check_card_delivery",
      "function_called": "check_card_delivery",
      "case_english": "Case: Check card delivery status.",
      "result": {
        "action": "check_card_delivery",
        "card_id": "K-9999",
        "carrier": "MockExpress",
        "eta": "1 days",
        "status": "in_transit",
        "last_update": "2025-08-31T08:27:32.243465Z"
      }
    }



```python

```
