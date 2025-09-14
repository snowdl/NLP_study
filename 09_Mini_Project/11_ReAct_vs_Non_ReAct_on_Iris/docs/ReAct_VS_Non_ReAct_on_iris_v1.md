```python
import sys, subprocess, shlex
print(sys.executable)  # 확인용
subprocess.check_call(shlex.split(f"{sys.executable} -m pip install --upgrade pip wikipedia"))
```

    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/bin/python
    Requirement already satisfied: pip in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (25.2)
    Requirement already satisfied: wikipedia in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (1.4.0)
    Requirement already satisfied: beautifulsoup4 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from wikipedia) (4.13.4)
    Requirement already satisfied: requests<3.0.0,>=2.0.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from wikipedia) (2.32.4)
    Requirement already satisfied: charset_normalizer<4,>=2 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from requests<3.0.0,>=2.0.0->wikipedia) (3.4.3)
    Requirement already satisfied: idna<4,>=2.5 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from requests<3.0.0,>=2.0.0->wikipedia) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from requests<3.0.0,>=2.0.0->wikipedia) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from requests<3.0.0,>=2.0.0->wikipedia) (2025.8.3)
    Requirement already satisfied: soupsieve>1.2 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from beautifulsoup4->wikipedia) (2.7)
    Requirement already satisfied: typing-extensions>=4.0.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages (from beautifulsoup4->wikipedia) (4.14.1)





    0




```python
import re
import time
import pandas as pd
import numpy as np
import wikipedia
```


```python
df = pd.read_csv("../11_data/iris.csv")
```


```python
print(df.head())
```

       sepal_length  sepal_width  petal_length  petal_width species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



```python
#React func
# Internal data queries are handled with pandas
#External knowledge queries are handled with the Wikipedia API.
```


```python
User input (query) ─┐
                    ├─> react_internal_avg_by_species(query, df)
DataFrame (df) ─────┘
                          │
                          ▼
                Dictionary (dict) returned
                ┌─────────────────────────────┐
                │ Thought: ...                 │
                │ Action: ...                  │
                │ Observation: pandas result   │
                │ Answer: final response text  │
                └─────────────────────────────┘
                          │
                          ▼
                   Stored in res variable

```


```python
# Execution Order (Always Follow These Steps)
"""
1) **Run data & tool cells**  
   Functions: `tool_mean_by_species`, `tool_wikipedia_summary`, `_infer_wiki_term`

2) **Run handler cells**  
   Functions: `baseline_internal`, `baseline_external`, `react_internal_avg_by_species`, `react_external_explain_species`

3) **Run evaluation utilities (4 cells)**  
   Functions: `eval_internal`, `eval_external`, `summarize` (and small helpers)

4) **Run examples or `run_all()`**  
   Produces on-screen results and saves CSVs (`react_vs_baseline_detail.csv`, `react_vs_baseline_summary.csv`).
   """
```




    '\n1) **Run data & tool cells**  \n   Functions: `tool_mean_by_species`, `tool_wikipedia_summary`, `_infer_wiki_term`\n\n2) **Run handler cells**  \n   Functions: `baseline_internal`, `baseline_external`, `react_internal_avg_by_species`, `react_external_explain_species`\n\n3) **Run evaluation utilities (4 cells)**  \n   Functions: `eval_internal`, `eval_external`, `summarize` (and small helpers)\n\n4) **Run examples or `run_all()`**  \n   Produces on-screen results and saves CSVs (`react_vs_baseline_detail.csv`, `react_vs_baseline_summary.csv`).\n   '




```python
#imports and helpers
#error handling
```


```python
from __future__ import annotations
from typing import Optional
import pandas as pd
```


```python
def _check_inputs(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Validate df and required columns; raise informative errors."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(column, str) or not column:
        raise TypeError("`column` must be a non-empty string.")

    # remove duplicates while preserving order
    required = ["species", column]
    seen = set()
    required_unique = [c for c in required if not (c in seen or seen.add(c))]

    missing = [c for c in required_unique if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}")
    return df
```


```python
df_ok = pd.DataFrame({
    "species": ["setosa","versicolor"],
    "petal_length": [1.4, 4.7],
})
print("_check_inputs OK →", _check_inputs(df_ok, "petal_length").equals(df_ok))

```

    _check_inputs OK → True



```python
# 2) no species 
df_no_species = pd.DataFrame({"petal_length":[1.4,4.7]})
try:
    _check_inputs(df_no_species, "petal_length")
except Exception as e:
    print("no species →", type(e).__name__, str(e))
```

    no species → ValueError Missing column(s): ['species']



```python
# 3) No target column 
df_no_col = pd.DataFrame({"species":["setosa","versicolor"]})
try:
    _check_inputs(df_no_col, "petal_length")
except Exception as e:
    print("no target column →", type(e).__name__, str(e))

```

    no target column → ValueError Missing column(s): ['petal_length']



```python
# 4)empty string case
try:
    _check_inputs(df_ok, "")
except Exception as e:
    print("empty column name →", type(e).__name__, str(e))
```

    empty column name → TypeError `column` must be a non-empty string.



```python
# 5)missing argument case
try:
    _check_inputs(df_ok) 
except Exception as e:
    print("missing argument →", type(e).__name__, str(e))
# 기대: TypeError _check_inputs() missing 1 required positional argument: 'column'
```

    missing argument → TypeError _check_inputs() missing 1 required positional argument: 'column'



```python

```


```python
# 6) column == 'species' check duplicates
try:
    _check_inputs(df_ok, "species")
    print("column == 'species' → OK (no duplicate missing names)")
except Exception as e:
    print("column == 'species' →", type(e).__name__, str(e))
```

    column == 'species' → OK (no duplicate missing names)



```python
df_ok = pd.DataFrame({"species":["setosa","virginica"], "petal_length":[1.4, 5.5]})
res = _prepare_df(df_ok, "petal_length", dropna=True)
print("✅ OK:", res.shape)
print(res)
```

    ✅ OK: (2, 2)
         species  petal_length
    0     setosa           1.4
    1  virginica           5.5



```python
import math
```


```python
def _prepare_df(df: pd.DataFrame, column: str, dropna: bool) -> pd.DataFrame:
    """Optionally drop NaNs and ensure the DataFrame is not empty afterwards."""
    work = df.dropna(subset=["species", column]) if dropna else df
    if work.empty:
        raise ValueError("No rows to aggregate after filtering (empty DataFrame).")
    return work
```


```python
df_err = pd.DataFrame({"species":[None, None], "petal_length":[math.nan, math.nan]})
try:
    _prepare_df(df_err, "petal_length", dropna=True)
    print("❌ Unexpected: no error")
except ValueError as e:
    print("✅ Raised as expected:", e)
```

    ✅ Raised as expected: No rows to aggregate after filtering (empty DataFrame).



```python
#------Tool function
"""
    Compute the mean of a numeric `column` grouped by the 'species' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain 'species' and the target `column`.
    column : str
        Name of the numeric column to aggregate (e.g., 'petal_length').
    dropna : bool, default True
        Drop rows where 'species' or the target column is NaN before grouping.
    digits : int or None, default None
        If provided, round the resulting means to this number of decimals.
    sort : bool, default True
        Sort the resulting Series by index (species).

    Returns
    -------
    pd.Series
        Series indexed by species with mean values.
        The Series name is set to f"mean_{column}_by_species".
"""
```




    '\n    Compute the mean of a numeric `column` grouped by the \'species\' column.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        DataFrame that must contain \'species\' and the target `column`.\n    column : str\n        Name of the numeric column to aggregate (e.g., \'petal_length\').\n    dropna : bool, default True\n        Drop rows where \'species\' or the target column is NaN before grouping.\n    digits : int or None, default None\n        If provided, round the resulting means to this number of decimals.\n    sort : bool, default True\n        Sort the resulting Series by index (species).\n\n    Returns\n    -------\n    pd.Series\n        Series indexed by species with mean values.\n        The Series name is set to f"mean_{column}_by_species".\n'




```python
def tool_mean_by_species(
    df: pd.DataFrame,
    column: str,
    *,
    dropna: bool = True,
    digits: Optional[int] = None,
    sort: bool = True,
) -> pd.Series:

    # validate & prepare
    work = _prepare_df(_check_inputs(df, column), column, dropna)

    # aggregate
    result = work.groupby("species")[column].mean()

    # post-process
    if digits is not None:
        result = result.round(digits)
    if sort:
        result = result.sort_index()

    result.name = f"mean_{column}_by_species"
    return result

```


```python
# Internaldata query
```


```python
result = tool_mean_by_species(df, "petal_length")
print("=== Average petal length by species ===")
print(result)
```

    === Average petal length by species ===
    species
    setosa        1.462
    versicolor    4.260
    virginica     5.552
    Name: mean_petal_length_by_species, dtype: float64



```python
#columns to test
columns_to_test = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
```


```python
for col in columns_to_test:
    result = tool_mean_by_species(df, col)
    print(f"\nColumn: {col}")
    print(result)
```

    
    Column: sepal_length
    species
    setosa        5.006
    versicolor    5.936
    virginica     6.588
    Name: mean_sepal_length_by_species, dtype: float64
    
    Column: sepal_width
    species
    setosa        3.428
    versicolor    2.770
    virginica     2.974
    Name: mean_sepal_width_by_species, dtype: float64
    
    Column: petal_length
    species
    setosa        1.462
    versicolor    4.260
    virginica     5.552
    Name: mean_petal_length_by_species, dtype: float64
    
    Column: petal_width
    species
    setosa        0.246
    versicolor    1.326
    virginica     2.026
    Name: mean_petal_width_by_species, dtype: float64



```python
from typing import Any, Dict
import re, time
import pandas as pd
```


```python
#helpers
```


```python
# Normalize user query: lowercase and collapse underscores/spaces into single spaces.
def _normalize_query(q: str) -> str:
    q = q.lower()
    return re.sub(r"[_\s]+", " ", q).strip()

# Check if query matches: average/avg/mean + (petal length | petal_length) + species
def _matches_avg_petal_by_species(q: str) -> bool:
    has_avg = any(k in q for k in ["average", "avg", "mean"])
    has_petal = any(k in q for k in ["petal length", "petal_length"])
    has_species = "species" in q
    return has_avg and has_petal and has_species
```


```python
#ReAct internal handler
```


```python
def react_internal_avg_by_species(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    t0 = time.perf_counter()
    qn = _normalize_query(query)

    # If the pattern matches, compute species-wise mean with the tool function.
    if _matches_avg_petal_by_species(qn):
        thought = "I need to calculate the average petal length by species."
        action = "pandas groupby → mean"
        try:
            # Dispatcher → call the actual tool (pandas aggregation)
            observation = tool_mean_by_species(df, "petal_length")
            # Compose a readable answer.
            answer = "Average petal length by species:\n" + observation.to_string()
            return {
                "Thought": thought, #Thought: human-readable reasoning step
                "Action": action, # Action:  chosen tool/operation (string label)
                "Observation": observation, # Observation: actual tool result (pd.Series)
                "Answer": answer, # Answer: final user-facing string
                "latency": time.perf_counter() - t0, #atency: wall-clock seconds for this call
                "tool_calls": 1, # tool_calls: how many tool ops were executed (for later comparison)
                "error": None, #error: error text if something failed, else None
            }

           # Surface a clean error message while preserving the ReAct schema.  
        except Exception as ex:
            return {
                "Thought": thought,
                "Action": action,
                "Observation": None,
                "Answer": f"Failed to compute means: {type(ex).__name__}: {ex}",
                "latency": time.perf_counter() - t0,
                "tool_calls": 1,
                "error": str(ex),
            }

    # Unmatched pattern
      # If the query doesn't match the pattern, return a gentle nudge.
    return {
        "Thought": "Pattern not matched",
        "Action": "None",
        "Observation": None,
        "Answer": "Try: 'average petal length by species'",
        "latency": time.perf_counter() - t0,
        "tool_calls": 0,
        "error": "unmatched",
    }
```


```python
# Wikipedia setup (install first with: pip install wikipedia)
```


```python
try:
    import wikipedia
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install `wikipedia` package (e.g., `pip install wikipedia`).")

# Set default language for the current session
wikipedia.set_lang("en")  # default; function below can override per-call
```


```python
from __future__ import annotations
from functools import lru_cache

# --- Tool function: actual executable code that calls Wikipedia ---
@lru_cache(maxsize=256)  # cache by (term, sentences, lang) to avoid repeat calls
def tool_wikipedia_summary(term: str, sentences: int = 2, lang: str = "en") -> str:
    """
    Return a short summary for `term` from Wikipedia.
    - Uses auto_suggest to correct common typos.
    - Handles disambiguation and missing-page errors gracefully.
    """
    try:
        # (0) Basic input sanitization
        term = (term or "").strip()
        if not term:
            return "Empty search term."

        # (1) Set Wikipedia language if provided (overrides the session default)
        if lang:
            wikipedia.set_lang(lang)

        # (2) Query Wikipedia summary
        return wikipedia.summary(
            term,
            sentences=max(1, int(sentences)),  # ensure >= 1
            auto_suggest=True
        )

    except wikipedia.DisambiguationError as e:   # (3-1) Ambiguity: multiple possible pages
        options_preview = ", ".join(e.options[:5])  # show first 5 candidates
        return f"Ambiguous term on Wikipedia. Please be more specific. Suggestions: {options_preview}"

    except wikipedia.PageError:                  # (3-2) Page not found
        return "No matching Wikipedia page found."

    except Exception as ex:                      # (3-3) Any other unexpected error
        return f"Wiki error: {type(ex).__name__}: {ex}"
```


```python
print(tool_wikipedia_summary("Iris setosa", sentences=2, lang="en")[:300])
print(tool_wikipedia_summary("Iris", sentences=1, lang="en")[:300])  # may hit disambiguation
```

    Iris setosa, known as the beachhead iris, bristle-pointed iris, or a number of other common names, is a species of flowering plant in the genus Iris of the family Iridaceae. It belongs the subgenus Limniris and the series Tripetalae.
    Ambiguous term on Wikipedia. Please be more specific. Suggestions: Ireland, Ireland, Éire, Erse (disambiguation), Republic of Ireland


    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages/wikipedia/wikipedia.py:389: GuessedAtParserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("html.parser"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    The code that caused this warning is on line 389 of the file /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_automl_env/lib/python3.10/site-packages/wikipedia/wikipedia.py. To get rid of this warning, pass the additional argument 'features="html.parser"' to the BeautifulSoup constructor.
    
      lis = BeautifulSoup(html).find_all('li')



```python
tool_wikipedia_summary("아이리스 세토사", lang="ko")  
```




    '청계천(淸溪川)은 대한민국 서울특별시 시내에 있는 지방하천으로, 한강 수계에 속하며 중랑천의 지류이다. 최장 발원지는 종로구 청운동에 위치한 ‘백운동 계곡’이며, 남으로 흐르다가 청계광장 부근의 지하에서 삼청동천을 합치며 몸집을 키운다.'




```python
tool_wikipedia_summary("Iris virginica", lang=None) 
```




    'No matching Wikipedia page found.'




```python
#Cell#2: External ReAct Execution Block
```


```python
from typing import Any, Dict
import time

# Normalize + pick a Wikipedia term from the query.
# If one of the three iris species is mentioned, map to the canonical page title.
def _infer_wiki_term(raw_query: str) -> str:
    q = (raw_query or "").strip().lower()
    if "setosa" in q:
        return "Iris setosa"
    if "versicolor" in q:
        return "Iris versicolor"
    if "virginica" in q:
        return "Iris virginica"
    return (raw_query or "").strip()  # fallback: use original input as-is
```


```python
  """
    ReAct-style external block:
      - Thought: reasoning text (not executed)
      - Action:  chosen tool label
      - Observation: tool output (Wikipedia summary string)
      - Answer: final user-facing response
      - latency/tool_calls: for later comparison
    """
```




    '\n  ReAct-style external block:\n    - Thought: reasoning text (not executed)\n    - Action:  chosen tool label\n    - Observation: tool output (Wikipedia summary string)\n    - Answer: final user-facing response\n    - latency/tool_calls: for later comparison\n  '




```python
def react_external_explain_species(query: str, lang: str = "en") -> Dict[str, Any]:
  
    t0 = time.perf_counter()

    # Decide which Wikipedia term to query (simple rule-based extraction).
    term = _infer_wiki_term(query)

    # Logs (for traceability; these are not executable steps).
    thought = f"Information about '{term}' is external; I'll query Wikipedia."
    action = "wikipedia.summary(term, sentences=2, lang)"

    # Execute the tool (Observation). tool_wikipedia_summary already handles errors.
    observation = tool_wikipedia_summary(term=term, sentences=2, lang=lang)

    # For now, the final answer is equal to the observation.
    answer = observation

    return {
        "Thought": thought,
        "Action": action,
        "Observation": observation,
        "Answer": answer,
        "latency": time.perf_counter() - t0,
        "tool_calls": 1,
        "error": None,
    }
```


```python
# Example 1: Normal case
res = react_external_explain_species("Explain Iris setosa", lang="en")
print("Thought:", res["Thought"])
print("Action:", res["Action"])
print("Observation:", res["Observation"])
print("Answer:", res["Answer"])

```

    Thought: Information about 'Iris setosa' is external; I'll query Wikipedia.
    Action: wikipedia.summary(term, sentences=2, lang)
    Observation: Iris setosa, known as the beachhead iris, bristle-pointed iris, or a number of other common names, is a species of flowering plant in the genus Iris of the family Iridaceae. It belongs the subgenus Limniris and the series Tripetalae.
    Answer: Iris setosa, known as the beachhead iris, bristle-pointed iris, or a number of other common names, is a species of flowering plant in the genus Iris of the family Iridaceae. It belongs the subgenus Limniris and the series Tripetalae.



```python
#non ReAct
"""
    Direct Wikipedia call without ReAct steps.
    Returns the same schema keys used for comparison (Answer/latency/tool_calls/error).
"""
```




    '\n    Direct Wikipedia call without ReAct steps.\n    Returns the same schema keys used for comparison (Answer/latency/tool_calls/error).\n'




```python
from typing import Any, Dict
import time

# Non-ReAct baseline: directly call the Wikipedia tool (no Thought/Action loop)
def baseline_external(query: str, lang: str = "en") -> Dict[str, Any]:
    """
    Direct Wikipedia call without ReAct steps.
    Adds success/note fields for easier reporting.
    """
    t0 = time.perf_counter()
    term = _infer_wiki_term(query)  # reuse the same mapper used by the ReAct version

    try:
        obs = tool_wikipedia_summary(term=term, sentences=2, lang=lang)
    except Exception as ex:
        # Safety net (tool_wikipedia_summary already handles most errors)
        return {
            "Answer": "",
            "Observation": None,
            "latency": time.perf_counter() - t0,
            "tool_calls": 1,
            "error": f"{type(ex).__name__}: {ex}",
            "success": 0.0,
            "note": "exception in baseline_external",
        }

    ok = _is_success_wiki(obs)
    return {
        "Answer": obs,
        "Observation": obs,
        "latency": time.perf_counter() - t0,
        "tool_calls": 1,
        "error": None if ok else "wiki_failure",
        "success": 1.0 if ok else 0.0,
        "note": "" if ok else obs[:120],
    }

# Simple success criterion for Wikipedia responses
_BAD_PREFIXES = ("Ambiguous term", "No matching Wikipedia page", "Wiki error")

def _is_success_wiki(answer: str, *, min_len: int = 30) -> bool:
    """Treat disambiguation/missing/error messages or very short replies as failures."""
    a = (answer or "").strip()
    if not a:
        return False
    if len(a) < min_len:
        return False
    # Safer than startswith (in case the library changes wording)
    if any(bad in a for bad in _BAD_PREFIXES):
        return False
    return True


```


```python
from typing import Any, Dict
import re, time
import pandas as pd

def baseline_internal(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Non-ReAct baseline for: 'average petal length by species'.
    - Robust matching with word-boundary regex (avg|average|mean), (petal[ _]?length), species
    - Single tool call via tool_mean_by_species
    - Stable return schema for later comparisons
    """
    t0 = time.perf_counter()

    # Normalize query once
    q = re.sub(r"[_\s]+", " ", (query or "").lower()).strip()

    # Word-boundary matching is safer than naive substring checks
    has_avg     = re.search(r"\b(avg|average|mean)\b", q) is not None
    has_p_length= re.search(r"\bpetal[ _]?length\b", q) is not None
    has_species = re.search(r"\bspecies\b", q) is not None

    if has_avg and has_p_length and has_species:
        try:
            obs = tool_mean_by_species(df, "petal_length")  # pandas groupby → mean
            ans = "Average petal length by species:\n" + obs.to_string()
            return {
                "Answer": ans,
                "Observation": obs,
                "latency": time.perf_counter() - t0,
                "tool_calls": 1,
                "error": None,
            }
        except Exception as ex:
            return {
                "Answer": f"Failed to compute means: {type(ex).__name__}: {ex}",
                "Observation": None,
                "latency": time.perf_counter() - t0,
                "tool_calls": 1,
                "error": str(ex),
            }

    # Pattern not matched → gentle nudge
    return {
        "Answer": "Pattern not matched. Try: 'average petal length by species'",
        "Observation": None,
        "latency": time.perf_counter() - t0,
        "tool_calls": 0,
        "error": "unmatched",
    }
```


```python
from typing import Any, Dict
import time

# Same simple success heuristic used in eval (treat disambiguation/missing/error as failures)
_BAD_PREFIXES = ("Ambiguous term", "No matching Wikipedia page", "Wiki error")

def _is_success_wiki(answer: str, *, min_len: int = 30) -> bool:
    """Treat disambiguation/missing/error messages or very short replies as failures."""
    a = (answer or "").strip()
    if not a or len(a) < min_len:
        return False
    return not any(bad in a for bad in _BAD_PREFIXES)

def baseline_external(query: str, lang: str = "en") -> Dict[str, Any]:
    """
    Non-ReAct baseline for external knowledge (Wikipedia).
    - Reuses the same term-mapper as ReAct to keep behavior consistent
    - Calls tool_wikipedia_summary once
    - Returns success/note fields for easier reporting
    """
    t0 = time.perf_counter()
    term = _infer_wiki_term(query)  # reuse the ReAct mapper (Iris setosa/versicolor/virginica → canonical)

    try:
        obs = tool_wikipedia_summary(term=term, sentences=2, lang=lang)
    except Exception as ex:
        # Safety net; tool_wikipedia_summary already handles typical errors
        return {
            "Answer": "",
            "Observation": None,
            "latency": time.perf_counter() - t0,
            "tool_calls": 1,
            "error": f"{type(ex).__name__}: {ex}",
            "success": 0.0,
            "note": "exception in baseline_external",
        }

    ok = _is_success_wiki(obs)
    return {
        "Answer": obs,
        "Observation": obs,
        "latency": time.perf_counter() - t0,
        "tool_calls": 1,
        "error": None if ok else "wiki_failure",
        "success": 1.0 if ok else 0.0,
        "note": "" if ok else obs[:120],
    }

```


```python
#check internal baseline 
```


```python
q = "average petal length by species"

for name, fn in [("baseline", baseline_internal), ("react", react_internal_avg_by_species)]:
    res = fn(q, df)
    print(f"[internal/{name}] tool_calls={res.get('tool_calls')}  latency_ms={res.get('latency',0)*1000:.2f}")
    print(res["Answer"], "\n")

```

    [internal/baseline] tool_calls=1  latency_ms=12.52
    Average petal length by species:
    species
    setosa        1.462
    versicolor    4.260
    virginica     5.552 
    
    [internal/react] tool_calls=1  latency_ms=2.53
    Average petal length by species:
    species
    setosa        1.462
    versicolor    4.260
    virginica     5.552 
    



```python
!which python
```


```python
import os
os.environ["MISTRAL_API_KEY"] = "MISTRAL_API_KEY"  # 네가 발급받은 실제 키
print(os.getenv("MISTRAL_API_KEY"))
```


```python
%cd /Users/jessicahong/11_data/open-rag-bench
!python -m openragbench.pipeline.data_processing.get_arxiv --help
```


```python
import sys
sys.path.append("/Users/jessicahong/11_data/open-rag-bench")  # 레포 루트
import openragbench
print("RAG Bench ready!", openragbench.__file__)

```

    RAG Bench ready! /Users/jessicahong/11_data/open-rag-bench/openragbench/__init__.py



```python
#Evalaution
```


```python
import time, pandas as pd, numpy as np

def _is_success_wiki(answer: str, min_len: int = 30) -> bool:
    bad = ("Ambiguous term", "No matching Wikipedia page", "Wiki error")
    a = (answer or "").strip()
    return bool(a) and len(a) >= min_len and not any(b in a for b in bad)

def eval_internal(df, queries, baseline_fn, react_fn):
    rows = []
    gt = df.groupby("species")["petal_length"].mean()
    for q in queries:
        # baseline
        b = baseline_fn(q, df)
        b_ok = isinstance(b.get("Observation"), pd.Series) and \
               b["Observation"].sort_index().equals(gt.sort_index())
        rows.append({"task":"internal","query":q,"method":"baseline",
                     "success":float(b_ok),"latency_ms":round(b["latency"]*1000,2),
                     "tool_calls":b.get("tool_calls",1)})
        # react
        r = react_fn(q, df)
        r_obs = r.get("Observation")
        r_ok = isinstance(r_obs, pd.Series) and r_obs.sort_index().equals(gt.sort_index())
        rows.append({"task":"internal","query":q,"method":"react",
                     "success":float(r_ok),"latency_ms":round(r["latency"]*1000,2),
                     "tool_calls":r.get("tool_calls",1)})
    return pd.DataFrame(rows)

def eval_external(queries, baseline_fn, react_fn, lang="en"):
    rows = []
    for q in queries:
        b = baseline_fn(q, lang=lang)
        rows.append({"task":"external","query":q,"method":"baseline",
                     "success":float(_is_success_wiki(b["Answer"])),
                     "latency_ms":round(b["latency"]*1000,2),
                     "tool_calls":b.get("tool_calls",1)})
        r = react_fn(q, lang=lang)
        rows.append({"task":"external","query":q,"method":"react",
                     "success":float(_is_success_wiki(r["Answer"])),
                     "latency_ms":round(r["latency"]*1000,2),
                     "tool_calls":r.get("tool_calls",1)})
    return pd.DataFrame(rows)

def summarize(df):
    return (df.groupby(["task","method"])
              .agg(success_rate=("success","mean"),
                   mean_latency_ms=("latency_ms","mean"),
                   mean_tool_calls=("tool_calls","mean"),
                   n=("success","count"))
              .round(3)
              .reset_index())

```


```python
# 내부 비교
qs_int = ["average petal length by species", "avg petal_length by species"]
detail_int = eval_internal(df, qs_int, baseline_internal, react_internal_avg_by_species)

# 외부 비교
qs_ext = ["Explain Iris setosa", "Explain Iris versicolor", "Explain Iris virginica", "Explain Iris flower"]
detail_ext = eval_external(qs_ext, baseline_external, react_external_explain_species, lang="en")

# 합치고 요약
detail = pd.concat([detail_int, detail_ext], ignore_index=True)
summary = summarize(detail)

print("=== Summary ===")
display(summary)
print("\n=== Detail (head) ===")
display(detail.head(10))

```

    === Summary ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task</th>
      <th>method</th>
      <th>success_rate</th>
      <th>mean_latency_ms</th>
      <th>mean_tool_calls</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>external</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>1499.202</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>external</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.008</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>internal</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>3.295</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>internal</td>
      <td>react</td>
      <td>1.0</td>
      <td>1.760</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Detail (head) ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task</th>
      <th>query</th>
      <th>method</th>
      <th>success</th>
      <th>latency_ms</th>
      <th>tool_calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>internal</td>
      <td>average petal length by species</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>4.70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>internal</td>
      <td>average petal length by species</td>
      <td>react</td>
      <td>1.0</td>
      <td>2.01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>internal</td>
      <td>avg petal_length by species</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>1.89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>internal</td>
      <td>avg petal_length by species</td>
      <td>react</td>
      <td>1.0</td>
      <td>1.51</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>external</td>
      <td>Explain Iris setosa</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>0.02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>external</td>
      <td>Explain Iris setosa</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>external</td>
      <td>Explain Iris versicolor</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>1993.94</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>external</td>
      <td>Explain Iris versicolor</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>external</td>
      <td>Explain Iris virginica</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>2046.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>external</td>
      <td>Explain Iris virginica</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Cell 1 — Utilities (imports & helpers)

import time, random
from typing import Callable, Dict, Any, List, Optional
import pandas as pd
import numpy as np

def _round_ms(seconds: float) -> float:
    """Convert seconds → milliseconds with rounding."""
    return round((seconds or 0.0) * 1000.0, 2)

def _series_equal(a: pd.Series, b: pd.Series, atol: float = 1e-9) -> bool:
    """Check equality of two Series: same index + close values."""
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        return False
    a = a.sort_index()
    b = b.sort_index()
    if not a.index.equals(b.index):
        return False
    try:
        return np.allclose(a.values, b.values, atol=atol, rtol=0.0, equal_nan=True)
    except Exception:
        return False

_BAD_PREFIXES = ("Ambiguous term", "No matching Wikipedia page", "Wiki error")

def _is_success_wiki(answer: str, min_len: int = 30) -> bool:
    """Treat disambiguation/missing/error or very short replies as failure."""
    a = (answer or "").strip()
    return bool(a) and len(a) >= min_len and not any(b in a for b in _BAD_PREFIXES)

```


```python
# Cell 2 — Internal evaluator

from typing import Tuple

def eval_internal(
    df: pd.DataFrame,
    queries: List[str],
    baseline_fn: Callable[[str, pd.DataFrame], Dict[str, Any]],
    react_fn: Callable[[str, pd.DataFrame], Dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate internal handlers against pandas ground truth."""
    gt = df.groupby("species")["petal_length"].mean()
    rows: List[Dict[str, Any]] = []

    for q in queries:
        # Baseline
        b = baseline_fn(q, df)
        b_ok = _series_equal(b.get("Observation"), gt)
        rows.append({
            "task": "internal",
            "query": q,
            "method": "baseline",
            "success": float(b_ok),
            "latency_ms": _round_ms(b.get("latency", 0.0)),
            "tool_calls": int(b.get("tool_calls", 1)),
        })

        # ReAct
        r = react_fn(q, df)
        r_ok = _series_equal(r.get("Observation"), gt)
        rows.append({
            "task": "internal",
            "query": q,
            "method": "react",
            "success": float(r_ok),
            "latency_ms": _round_ms(r.get("latency", 0.0)),
            "tool_calls": int(r.get("tool_calls", 1)),
        })

    return pd.DataFrame(rows)
```


```python
# Cell 3 — External evaluator (fair mode / repeats / shuffling)

def eval_external(
    queries: List[str],
    baseline_fn: Callable[..., Dict[str, Any]],
    react_fn: Callable[..., Dict[str, Any]],
    *,
    lang: str = "en",
    fair: bool = False,
    repeats: int = 1,
    shuffle: bool = False,
    cache_clear_fn: Optional[Callable[[], None]] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Evaluate external (Wikipedia) handlers.

    fair=True  → clear tool cache before each call (avoids 'first caller pays' bias)
    repeats>1  → run multiple times per (query, method) and average metrics
    shuffle=True → randomize method order per query to reduce order effects
    cache_clear_fn → typically: getattr(tool_wikipedia_summary, "cache_clear", None)
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    def _maybe_clear():
        if fair and cache_clear_fn is not None:
            try:
                cache_clear_fn()
            except Exception:
                pass

    for q in queries:
        methods = [("baseline", baseline_fn), ("react", react_fn)]
        if shuffle:
            rng.shuffle(methods)

        for name, fn in methods:
            latencies, successes, toolcalls = [], [], []
            for _ in range(max(1, repeats)):
                _maybe_clear()
                res = fn(q, lang=lang)  # both baseline_external / react_external take (query, lang=)
                ans = res.get("Answer", "")
                succ = res.get("success")
                if succ is None:
                    succ = float(_is_success_wiki(ans))
                latencies.append(_round_ms(res.get("latency", 0.0)))
                successes.append(float(succ))
                toolcalls.append(int(res.get("tool_calls", 1)))

            rows.append({
                "task": "external",
                "query": q,
                "method": name,
                "success": float(np.mean(successes)),
                "latency_ms": float(np.mean(latencies)),
                "tool_calls": float(np.mean(toolcalls)),
                "repeats": repeats,
                "fair": fair,
                "shuffled": shuffle,
            })

    return pd.DataFrame(rows)
```


```python
# Cell 4 — Summary

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by task/method into success rate, mean latency, mean tool calls."""
    out = (df.groupby(["task","method"])
             .agg(success_rate=("success","mean"),
                  mean_latency_ms=("latency_ms","mean"),
                  mean_tool_calls=("tool_calls","mean"),
                  n=("success","count"))
             .round(3)
             .reset_index())
    return out

# --- Example usage (uncomment to run) ---
# # Internal
# detail_int = eval_internal(
#     df=df,
#     queries=["average petal length by species", "avg petal_length by species"],
#     baseline_fn=baseline_internal,
#     react_fn=react_internal_avg_by_species,
# )
#
# # External (fair mode: clear LRU cache each call, shuffle order, 1 repeat)
# detail_ext = eval_external(
#     queries=["Explain Iris setosa","Explain Iris versicolor","Explain Iris virginica","Explain Iris flower"],
#     baseline_fn=baseline_external,
#     react_fn=react_external_explain_species,
#     lang="en",
#     fair=True,
#     repeats=1,
#     shuffle=True,
#     cache_clear_fn=getattr(tool_wikipedia_summary, "cache_clear", None),
# )
#
# detail = pd.concat([detail_int, detail_ext], ignore_index=True)
# display(summarize(detail))
# display(detail.head(10))
```


```python
summarize(detail)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task</th>
      <th>method</th>
      <th>success_rate</th>
      <th>mean_latency_ms</th>
      <th>mean_tool_calls</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>external</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>1499.202</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>external</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.008</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>internal</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>3.295</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>internal</td>
      <td>react</td>
      <td>1.0</td>
      <td>1.760</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
detail.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task</th>
      <th>query</th>
      <th>method</th>
      <th>success</th>
      <th>latency_ms</th>
      <th>tool_calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>internal</td>
      <td>average petal length by species</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>4.70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>internal</td>
      <td>average petal length by species</td>
      <td>react</td>
      <td>1.0</td>
      <td>2.01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>internal</td>
      <td>avg petal_length by species</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>1.89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>internal</td>
      <td>avg petal_length by species</td>
      <td>react</td>
      <td>1.0</td>
      <td>1.51</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>external</td>
      <td>Explain Iris setosa</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>0.02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>external</td>
      <td>Explain Iris setosa</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>external</td>
      <td>Explain Iris versicolor</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>1993.94</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>external</td>
      <td>Explain Iris versicolor</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>external</td>
      <td>Explain Iris virginica</td>
      <td>baseline</td>
      <td>1.0</td>
      <td>2046.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>external</td>
      <td>Explain Iris virginica</td>
      <td>react</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
