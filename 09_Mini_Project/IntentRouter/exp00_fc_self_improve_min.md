```python
#Imports & Config
```


```python
import re
from typing import Dict, List, Tuple, Any, Optional

THRESHOLD = 1  # accept intent if we have >= 1 keyword hit
```


```python
#Normalizer
```


```python
# Text normalizer:
# - Convert to lowercase
# - Remove symbols/punctuation (keep letters, numbers, Korean)
# - Collapse multiple spaces into one
_WS = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    return _WS.sub(" ", s).strip()
```


```python
# === Quick check for normalize_text ===
samples = [
    "Hello, WORLD!!!",          # English with punctuation
    "Card stolen!!! 123",       # Mixed with numbers
    "분실 신고 해주세요!!!",       # Korean with punctuation
    "Multiple     spaces here", # Extra spaces
]

for s in samples:
    print(f"Input : {s}")
    print(f"Output: {normalize_text(s)}")
    print("-" * 40)
```

    Input : Hello, WORLD!!!
    Output: hello world
    ----------------------------------------
    Input : Card stolen!!! 123
    Output: card stolen 123
    ----------------------------------------
    Input : 분실 신고 해주세요!!!
    Output: 분실 신고 해주세요
    ----------------------------------------
    Input : Multiple     spaces here
    Output: multiple spaces here
    ----------------------------------------



```python
#Rules (in-code)
```


```python
# === Rules: intents and their keywords ===
# - Each intent has a list of keywords that may appear in user text
# - "args" tells us what parameters are required (user_id or card_id)

RULES: Dict[str, Dict[str, List[str]]] = {
    "report_card_lost": {
        "keywords": [
            "lost my card", "card stolen", "card missing",
            "misplaced", "lost", "missing card"
        ],
        "args": ["user_id"],
    },
    "check_card_delivery": {
        "keywords": [
            "track my card", "where is my card", "card not arrived",
            "delivered", "delivery", "card delivery", "when will my card arrive"
        ],
        "args": ["card_id"],
    },
    "request_new_card": {
        "keywords": [
            "order new card", "replace my card", "new card",
            "replacement", "get replacement", "physical card"
        ],
        "args": ["user_id"],
    },
}
```


```python
#Mock action handlers
```


```python
def report_card_lost(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Simulate reporting a lost card.
    Args:
        user_id (str, optional): ID of the user who lost the card.
    Returns:
        dict: result with ok flag, fake ticket_id, and user_id
    """
    return {"ok": True, "ticket_id": "T12345", "user_id": user_id}
```


```python
print("Test report_card_lost:")
print(report_card_lost("U123"))
```

    Test report_card_lost:
    {'ok': True, 'ticket_id': 'T12345', 'user_id': 'U123'}



```python
def check_card_delivery(card_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Simulate checking the delivery status of a card.
    Args:
        card_id (str, optional): ID of the card being tracked.
    Returns:
        dict: result with ok flag, card_id, carrier, eta, and status
    """
    return {
        "ok": True,
        "card_id": card_id,
        "carrier": "UPS",
        "eta": "3-5 business days",
        "last_update": "out for delivery",
    }
```


```python
print("\nTest check_card_delivery:")
print(check_card_delivery("C5555"))
```

    
    Test check_card_delivery:
    {'ok': True, 'card_id': 'C5555', 'carrier': 'UPS', 'eta': '3-5 business days', 'last_update': 'out for delivery'}



```python
def request_new_card(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Simulate placing a new card order.
    Args:
        user_id (str, optional): ID of the user requesting a new card.
    Returns:
        dict: result with ok flag, fake order_id, and user_id
    """
    return {"ok": True, "order_id": "O98765", "user_id": user_id}
```


```python
print("\nTest request_new_card:")
print(request_new_card("U999"))
```

    
    Test request_new_card:
    {'ok': True, 'order_id': 'O98765', 'user_id': 'U999'}



```python
#Intent detector (count hits only)
"""
    Predict the intent of a given text based on keyword rules.

    Args:
        text (str): User input message.
        rules (dict): Dictionary of intents. Each intent contains:
                      - "keywords": list of strings to match
                      - "args": list of required arguments (optional)

    Returns:
        (intent, hits):
            - intent (str): name of the best-matching intent,
                            or "nlu_fallback" if no keywords matched
            - hits (int): how many keywords matched
"""
```




    '\n    Predict the intent of a given text based on keyword rules.\n\n    Args:\n        text (str): User input message.\n        rules (dict): Dictionary of intents. Each intent contains:\n                      - "keywords": list of strings to match\n                      - "args": list of required arguments (optional)\n\n    Returns:\n        (intent, hits):\n            - intent (str): name of the best-matching intent,\n                            or "nlu_fallback" if no keywords matched\n            - hits (int): how many keywords matched\n'




```python
def predict_intent(text: str, rules: Dict[str, Dict]) -> Tuple[str, int]:

    # Normalize text and pad with spaces for safe matching
    t = " " + normalize_text(text) + " "

    # Default = fallback intent (no matches)
    best = ("nlu_fallback", 0)

    # Loop through each intent and count keyword matches
    for intent, cfg in rules.items():
        hits = 0
        for kw in cfg.get("keywords", []):
            # Normalize each keyword and check if it's in the text
            if kw.strip() and (" " + normalize_text(kw) + " ") in t:
                hits += 1
        # Keep the intent with the highest number of matches
        if hits > best[1]:
            best = (intent, hits)

    return best

```


```python
# === Quick test for predict_intent ===

# Example rules
rules = {
    "report_card_lost": {
        "keywords": ["lost my card", "card stolen", "card missing", "misplaced"],
    },
    "check_card_delivery": {
        "keywords": ["track my card", "where is my card", "card not arrived", "delivery"],
    },
    "request_new_card": {
        "keywords": ["order new card", "replace my card", "new card", "replacement"],
    },
}

# Example test sentences
tests = [
    "I lost my card yesterday",
    "Can you track my card delivery?",
    "Please order a new card for me",
    "When will my card be delivered?",
    "What is the weather today?"  # should fallback
]

# Run predictions
for t in tests:
    intent, hits = predict_intent(t, rules)
    print(f"Input: {t}")
    print(f" → Predicted intent: {intent}, keyword matches={hits}\n")

```

    Input: I lost my card yesterday
     → Predicted intent: report_card_lost, keyword matches=1
    
    Input: Can you track my card delivery?
     → Predicted intent: check_card_delivery, keyword matches=2
    
    Input: Please order a new card for me
     → Predicted intent: request_new_card, keyword matches=1
    
    Input: When will my card be delivered?
     → Predicted intent: nlu_fallback, keyword matches=0
    
    Input: What is the weather today?
     → Predicted intent: nlu_fallback, keyword matches=0
    



```python
#Routing helpers
```


```python
# Block 6: handler map + small helper for required args
INTENT_TO_FUNC = {
    "report_card_lost": report_card_lost,
    "check_card_delivery": check_card_delivery,
    "request_new_card": request_new_card,
}

def required_args_present(intent: str, need: List[str], user_id: Optional[str], card_id: Optional[str]) -> Tuple[bool, List[str]]:
    bag = {"user_id": user_id, "card_id": card_id}
    missing = [k for k in need if bag.get(k) is None]
    return (len(missing) == 0, missing)
```


```python
#Pipeline
"""
    End-to-end pipeline for handling user requests.

    Steps:
    1. Detect the intent of the input text using predict_intent().
    2. If no intent (hits < THRESHOLD), return fallback message.
    3. If intent is found, look up the matching handler function.
    4. Check if required arguments (user_id, card_id) are provided.
    5. If arguments are missing, ask the user to provide them.
    6. Otherwise, call the handler and return its result.
"""
```




    '\n    End-to-end pipeline for handling user requests.\n\n    Steps:\n    1. Detect the intent of the input text using predict_intent().\n    2. If no intent (hits < THRESHOLD), return fallback message.\n    3. If intent is found, look up the matching handler function.\n    4. Check if required arguments (user_id, card_id) are provided.\n    5. If arguments are missing, ask the user to provide them.\n    6. Otherwise, call the handler and return its result.\n'




```python
# === run_pipeline — detect → fallback → call handler ===
def run_pipeline(text: str, user_id: Optional[str] = None, card_id: Optional[str] = None) -> Dict[str, Any]:
    # Step 1: detect intent
    intent, hits = predict_intent(text, RULES)

    # Step 2: handle fallback (low confidence)
    if hits < THRESHOLD:
        return {
            "message": "Sorry, I couldn't understand. Please rephrase.",
            "intent": "nlu_fallback",
            "hits": hits,
            "called": None,
            "result": None,
        }

    # Step 3: find the handler function for the detected intent
    fn = INTENT_TO_FUNC.get(intent)
    if not fn:
        return {
            "message": f"Detected intent '{intent}' but no handler is set.",
            "intent": intent,
            "hits": hits,
            "called": None,
            "result": None,
        }

    # Step 4: check if required arguments are provided
    need = RULES[intent].get("args", [])
    ok, missing = required_args_present(intent, need, user_id, card_id)
    if not ok:
        return {
            "message": f"Please provide your {', '.join(missing)}.",
            "intent": intent,
            "hits": hits,
            "called": None,
            "result": None,
        }

    # Step 5: prepare kwargs and call the handler
    kwargs = {}
    if "user_id" in need:
        kwargs["user_id"] = user_id
    if "card_id" in need:
        kwargs["card_id"] = card_id
    res = fn(**kwargs)

    # Step 6: return success response
    return {
        "message": f"✅ {intent} done.",
        "intent": intent,
        "hits": hits,
        "called": fn.__name__,
        "result": res,
    }

```


```python
#Tiny smoke test
```


```python
tests = [
    {"text": "I lost my card yesterday", "user_id": "U1"},               # lost card
    {"text": "Where is my card? It hasn't arrived.", "card_id": "C1"},   # delivery check
    {"text": "Please order a new card", "user_id": "U2"},                # new card request
    {"text": "My card went missing last night", "user_id": "U3"},        # lost card (alt phrasing)
    {"text": "When will my card be delivered?", "card_id": "C2"},        # delivery check (alt phrasing)
    {"text": "Hello, can you help me?"},                                 # fallback expected
]

for t in tests:
    out = run_pipeline(t["text"], user_id=t.get("user_id"), card_id=t.get("card_id"))
    print(t["text"], "->", out)
```

    I lost my card yesterday -> {'message': '✅ report_card_lost done.', 'intent': 'report_card_lost', 'hits': 2, 'called': 'report_card_lost', 'result': {'ok': True, 'ticket_id': 'T12345', 'user_id': 'U1'}}
    Where is my card? It hasn't arrived. -> {'message': '✅ check_card_delivery done.', 'intent': 'check_card_delivery', 'hits': 1, 'called': 'check_card_delivery', 'result': {'ok': True, 'card_id': 'C1', 'carrier': 'UPS', 'eta': '3-5 business days', 'last_update': 'out for delivery'}}
    Please order a new card -> {'message': '✅ request_new_card done.', 'intent': 'request_new_card', 'hits': 1, 'called': 'request_new_card', 'result': {'ok': True, 'order_id': 'O98765', 'user_id': 'U2'}}
    My card went missing last night -> {'message': "Sorry, I couldn't understand. Please rephrase.", 'intent': 'nlu_fallback', 'hits': 0, 'called': None, 'result': None}
    When will my card be delivered? -> {'message': '✅ check_card_delivery done.', 'intent': 'check_card_delivery', 'hits': 1, 'called': 'check_card_delivery', 'result': {'ok': True, 'card_id': 'C2', 'carrier': 'UPS', 'eta': '3-5 business days', 'last_update': 'out for delivery'}}
    Hello, can you help me? -> {'message': "Sorry, I couldn't understand. Please rephrase.", 'intent': 'nlu_fallback', 'hits': 0, 'called': None, 'result': None}



```python

```
