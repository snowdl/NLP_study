```python
from pathlib import Path                     # OS-agnostic file path handling
import json, hashlib, statistics as stats    # JSON I/O, content hash IDs, simple descriptive stats
import re, unicodedata                       # Regex ops + Unicode normalization/whitespace fixes
import nltk                                  # Sentence tokenization (Punkt)
import tiktoken                              # GPT-style tokenizer for token counting
```


```python
# ========= Paths / Settings =========
DOC_NAME = "01 Harry Potter and the Sorcerers Stone.txt"   # Source filename
DOC_PATH = Path("../../11_data") / DOC_NAME                # Full path to the input text
OUT_JSONL = Path("hp_chunks_100tok.jsonl")                 # Output: all chunks with id/cid/tokens/text
OUT_JSONL_DEDUP = Path("hp_chunks_100tok.dedup.jsonl")     # Output: exact-deduplicated chunks
MAX_TOKENS = 100                                           # Target max tokens per chunk (sentence-safe)
OVERLAP_TOKENS = 10                                        # Token overlap between adjacent chunks (0 = off)

```


```python
# ========= 전처리 =========
"""
    Text cleaning for OCR/Unicode artifacts and broken honorifics.
    - Normalizes Unicode width/compatibility.
    - Flattens invisible/zero-width/nbsp-like spaces.
    - Repairs hyphen line-breaks and newline spacing.
    - Fixes broken 'M r.' → 'Mr.' and ensures a space after 'Mr.' when needed.
    - Canonicalizes common honorifics (Mrs., Ms., Dr., Prof.).
    - Collapses 'H .' → 'H.' for single-letter initials.
    Returns a cleaned string; does NOT alter semantics.
 """
def clean_text(text: str) -> str:
    # Normalize to NFKC so full-width/compatibility forms (quotes, spaces, etc.) are unified
    t = unicodedata.normalize("NFKC", text)

    # Normalize newlines to '\n'
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Replace common zero-width/nbsp characters with a plain space (prevents tokenization glitches)
    t = re.sub(r"[\u00A0\u200B\u200C\u200D]", " ", t)  # NBSP & zero-width variants → " "
    t = re.sub(r"[ \t]{2,}", " ", t)                  # Collapse multiple ASCII spaces/tabs

    # Join hyphen line-breaks: "some-\nthing" → "something"
    t = re.sub(r"-\s*\n\s*", "", t)

    # Paragraph-aware newline handling:
    # - Keep double newlines as paragraph breaks
    # - Turn single newlines into a single space
    t = re.sub(r"\n{2,}", "\n\n", t)                  # Preserve paragraphs
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)            # Inline single newline → space
    t = re.sub(r"[ \t]{2,}", " ", t).strip()          # Final whitespace squeeze

    # --- Generic 'Mr.' repair (covers hidden spaces/quotes/parens around it) ---
    # Optional opening quote/paren + 'm' + any zero-width/space + 'r' + '.' + optional closing quote/paren
    # Case-insensitive; preserves surrounding punctuation via capture groups.
    t = re.sub(
        r'(?i)(["“‘\'(\[]?\s*)m[\s\u00A0\u200B\u200C\u200D]*r[\s\u00A0\u200B\u200C\u200D]*\.(\s*["”’\'\])]?)+',
        r'\1Mr.\2',
        t
    )

    # --- B patch 1) Fix ONLY true 'M r.' cases (requires at least 1 whitespace-like char between M and r) ---
    # This avoids touching already-correct 'Mr.' but heals OCR splits like "M r."
    t = re.sub(
        r'(?i)(["“‘\'(\[]?\s*)m[\s\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000\uFEFF\u200B\u200C\u200D\u2060]+r\s*\.(\s*["”’\'\])]?)+',
        r'\1Mr.\2',
        t
    )

    # --- B patch 2) Ensure a space after titles when followed by a letter ---
    # e.g., "Mr.Dursley" / "Mr.and" → "Mr. Dursley" / "Mr. and"
    t = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof)\.(?=[A-Za-z])', r'\1. ', t)

    # --- Canonicalize other honorifics (robust to stray spaces): ---
    # \b m s* rs s* \. \b → "Mrs." etc., case-insensitive
    abbrev_patterns = {
        r"\bm\s*rs\s*\.\b": "Mrs.",
        r"\bm\s*s\s*\.\b":  "Ms.",
        r"\bd\s*r\s*\.\b":  "Dr.",
        r"\bp\s*rof\s*\.\b": "Prof.",
    }
    for pat, rep in abbrev_patterns.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)

    # --- Initials: "H ." → "H." (handles hidden/zero-width spaces as well) ---
    t = re.sub(r"(?i)(?<![A-Za-z])([A-Z])[\s\u00A0\u200B\u200C\u200D]+\.", r"\1.", t)

    return t

```


```python
# ========= Tokenizer =========
try:
    # Prefer a model-specific tokenizer so counts match GPT-4 behavior
    enc = tiktoken.encoding_for_model("gpt-4")
except Exception:
    # Fallback: generic CL100K encoding (compatible with GPT-3.5/4 families)
    enc = tiktoken.get_encoding("cl100k_base")

def tok_len(text: str) -> int:
    """Return the token count for `text` using the selected `enc` tokenizer."""
    return len(enc.encode(text))

```


```python
# ========= Chunking function (never split inside a sentence) =========
def chunk_by_tokens_sentence_safe(sents, max_tokens=100, overlap_tokens=0):
    """
    Build token-limited chunks while preserving sentence boundaries.

    Behavior:
    - Append sentences until adding the next one would exceed `max_tokens`,
      then flush the current chunk and start a new one.
    - If a *single* sentence already exceeds `max_tokens`, make it a
      standalone chunk (allowed to exceed the limit) to avoid splitting the sentence.
    - If `overlap_tokens` > 0, copy the last N tokens from the previous chunk
      to the start of the next chunk (context carry-over).

    Args:
        sents (List[str]): Pre-tokenized sentences (already cleaned).
        max_tokens (int): Target max tokens per chunk.
        overlap_tokens (int): Number of tail tokens to overlap between chunks.

    Returns:
        List[Dict[str, Any]]: Each item is {"text": str, "tokens": int}.
    """
    chunks = []                  # Accumulated output chunks
    cur_texts, cur_tokens = [], 0  # Current chunk: sentence list + token count

    def flush():
        """Finalize the current chunk (if non-empty) and reset buffers."""
        nonlocal cur_texts, cur_tokens
        if not cur_texts:
            return
        text = " ".join(cur_texts).strip()           # Join sentences with a space
        chunks.append({"text": text, "tokens": cur_tokens})
        cur_texts, cur_tokens = [], 0                # Reset for the next chunk

    for s in sents:
        n = tok_len(s)  # Token length of the incoming sentence

        # Case 1: Single sentence longer than the limit → make its own chunk.
        # We never split inside a sentence to keep semantics intact.
        if n > max_tokens:
            flush()                                   # Close any current chunk
            chunks.append({"text": s.strip(), "tokens": n})
            continue

        # Case 2: Adding this sentence would exceed the limit → flush first.
        if cur_tokens > 0 and (cur_tokens + n > max_tokens):
            flush()

            # Optional overlap: prepend the last `overlap_tokens` tokens
            # from the *previous* chunk to the new one for continuity.
            if overlap_tokens > 0 and len(chunks) > 0:
                tail_text = chunks[-1]["text"]            # Previous chunk text
                tail_ids = enc.encode(tail_text)          # Token IDs of tail
                ov_ids = tail_ids[max(0, len(tail_ids) - overlap_tokens):]
                ov_text = enc.decode(ov_ids).strip()      # Overlap text to seed
                cur_texts = [ov_text] if ov_text else []  # Start next chunk with overlap
                # Recompute token count from the overlap text (if any)
                cur_tokens = len(enc.encode(" ".join(cur_texts))) if cur_texts else 0

        # Case 3: Safe to add the sentence to the current chunk.
        cur_texts.append(s)
        cur_tokens += n

    # Flush the final (possibly partial) chunk.
    flush()
    return chunks

```


```python
# ========= Chunking helpers =========
def _overlap_seed(prev_text: str, overlap_tokens: int, enc) -> tuple[str, int]:
    """
    Build an overlap seed from the tail of `prev_text`.
    Returns (seed_text, seed_token_len). Empty seed if overlap_tokens <= 0.
    """
    if overlap_tokens <= 0 or not prev_text:
        return "", 0
    ids = enc.encode(prev_text)
    if not ids:
        return "", 0
    ov_ids = ids[max(0, len(ids) - overlap_tokens):]
    seed_text = enc.decode(ov_ids).strip()
    seed_tokens = len(enc.encode(seed_text)) if seed_text else 0
    return seed_text, seed_tokens


def _flush(cur_texts: list[str], cur_tokens: int, chunks: list[dict]) -> tuple[list[str], int]:
    """
    Append current buffer to `chunks` (if non-empty) and reset the buffer.
    Returns (new_cur_texts, new_cur_tokens).
    """
    if cur_texts:
        text = " ".join(cur_texts).strip()
        chunks.append({"text": text, "tokens": cur_tokens})
    return [], 0


# ========= Chunking function (never split inside a sentence) =========
def chunk_by_tokens_sentence_safe(
    sents: list[str],
    max_tokens: int = 100,
    overlap_tokens: int = 0,
    enc=enc,
) -> list[dict]:
    """
    Build token-limited chunks while preserving sentence boundaries.

    Rules:
    - Keep appending sentences until adding the next would exceed `max_tokens`,
      then flush the current chunk.
    - If a single sentence itself exceeds `max_tokens`, emit it as a standalone chunk
      (do NOT split inside that sentence).
    - If `overlap_tokens` > 0, copy the last N tokens of the previous chunk
      to the head of the next chunk as context.

    Args:
        sents: pre-cleaned sentences.
        max_tokens: target upper bound per chunk (soft; long sentences may exceed).
        overlap_tokens: tail tokens to overlap between adjacent chunks.
        enc: tokenizer (tiktoken encoding).

    Returns:
        List of {"text": str, "tokens": int}.
    """
    chunks: list[dict] = []
    cur_texts: list[str] = []
    cur_tokens: int = 0

    for s in sents:
        n = len(enc.encode(s))

        # Case 1: single very long sentence → standalone chunk
        if n > max_tokens:
            cur_texts, cur_tokens = _flush(cur_texts, cur_tokens, chunks)
            chunks.append({"text": s.strip(), "tokens": n})
            continue

        # Case 2: adding this sentence would exceed the limit → flush then start new chunk
        if cur_tokens > 0 and (cur_tokens + n > max_tokens):
            cur_texts, cur_tokens = _flush(cur_texts, cur_tokens, chunks)

            # Optional overlap from the previous chunk
            if overlap_tokens > 0 and chunks:
                seed_text, seed_tokens = _overlap_seed(chunks[-1]["text"], overlap_tokens, enc)
                if seed_text:
                    cur_texts = [seed_text]
                    cur_tokens = seed_tokens

        # Case 3: safe to append sentence
        cur_texts.append(s)
        cur_tokens += n

    # Flush the trailing buffer
    _ = _flush(cur_texts, cur_tokens, chunks)
    return chunks


# (Optional) Generator version if you prefer streaming:
def iter_chunks_sentence_safe(
    sents: list[str],
    max_tokens: int = 100,
    overlap_tokens: int = 0,
    enc=enc,
):
    """
    Yield chunks one by one (useful for large corpora / streaming).
    Behavior matches `chunk_by_tokens_sentence_safe`.
    """
    chunks: list[dict] = []
    cur_texts: list[str] = []
    cur_tokens: int = 0

    def flush_yield():
        nonlocal cur_texts, cur_tokens
        if cur_texts:
            text = " ".join(cur_texts).strip()
            yield {"text": text, "tokens": cur_tokens}
            cur_texts, cur_tokens = [], 0

    for s in sents:
        n = len(enc.encode(s))
        if n > max_tokens:
            # Flush current and yield long sentence directly
            yield from flush_yield()
            yield {"text": s.strip(), "tokens": n}
            continue

        if cur_tokens > 0 and (cur_tokens + n > max_tokens):
            # Flush current
            yield from flush_yield()
            # Overlap
            if overlap_tokens > 0 and chunks:
                seed_text, seed_tokens = _overlap_seed(chunks[-1]["text"], overlap_tokens, enc)
                if seed_text:
                    cur_texts = [seed_text]
                    cur_tokens = seed_tokens

        # Append sentence
        cur_texts.append(s)
        cur_tokens += n

        # Keep a shadow copy of the last emitted chunk for overlap seeding
        # (only updated when we would flush/yield)
        # We simulate this by appending to `chunks` when we yield.
        # To keep it simple, we only push into `chunks` when we actually yield:
        # so we mirror `chunks` here:
        # (No-op now; we'll update `chunks` in the flush below.)

        # When we might yield, we cache the to-be-emitted chunk:
        # We do this by peeking at the buffer. Not necessary per-yield.# ========= Hash (ID) =========
def cid16(text: str) -> str:
    """Content-based ID: first 16 hex chars of the SHA-256 digest (~64-bit prefix)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


        # NOTE: After yield, we append to `chunks` to preserve overlap history.
        # This block is handled below.

        # If you'd like strict streaming with no extra memory, drop overlap or
        # store only the last-emitted chunk text externally.

        # (no-op here)

        # If the next sentence causes a flush, the overlap will use the last yielded chunk.

    # Final flush
    if cur_texts:
        text = " ".join(cur_texts).strip()
        last = {"text": text, "tokens": cur_tokens}
        yield last

```


```python
# ========= Hash (ID) =========
def cid16(text: str) -> str:
    """Content-based ID: first 16 hex chars of the SHA-256 digest (~64-bit prefix)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
```


```python

```


```python
# ========= 메인 =========

from typing import List, Dict, Optional, Tuple

def ensure_punkt() -> None:
    """Make sure NLTK 'punkt' tokenizer data is available."""
    try:
        # Check if the tokenizer model is already installed
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        # Download 'punkt' once if missing
        nltk.download("punkt")


def load_text_or_none(path: Path) -> Optional[str]:
    """Read UTF-8 text from `path`; print a friendly error and return None if missing."""
    if not path.exists():
        # User-friendly message if the file path is wrong
        print(f"[오류] 파일을 찾을 수 없습니다: {path.resolve()}")
        print("→ DOC_PATH 경로를 확인하거나 파일을 해당 위치로 옮겨주세요.")
        return None
    # Read the file as UTF-8; ignore undecodable bytes
    return path.read_text(encoding="utf-8", errors="ignore")


def preprocess_and_sentence_split(text: str) -> List[str]:
    """Apply cleaning then split into sentences."""
    # Normalize/fix text artifacts first (improves sentence tokenization)
    text = clean_text(text)
    # Split the cleaned text into sentences with NLTK
    return nltk.sent_tokenize(text)


def build_chunks_from_sents(sents: List[str]) -> List[Dict[str, int | str]]:
    """Chunk sentences with token budget and optional overlap (never split inside a sentence)."""
    # Delegate to the sentence-safe chunker; uses global MAX_TOKENS/OVERLAP_TOKENS
    return chunk_by_tokens_sentence_safe(
        sents, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS
    )


def summarize_chunks(chunks: List[Dict[str, int | str]]) -> List[int]:
    """Print summary stats and return token lengths."""
    # Collect token lengths for descriptive statistics
    lens = [c["tokens"] for c in chunks]

    print("=== CHUNKS SUMMARY ===")
    print(f"총 청크 수: {len(chunks)}")
    print(
        "토큰수(평균/중앙/최소/최대): "
        f"{round(stats.mean(lens),2)} / {stats.median(lens)} / {min(lens)} / {max(lens)}"
    )

    # Share of chunks that are close to the budget (90–100 tokens)
    pct_90_100 = round(sum(1 for x in lens if 90 <= x <= 100) / len(lens) * 100, 2)
    print(f"100 토큰 근접(90~100) 비율: {pct_90_100}%")

    # Show the three longest chunks for quick inspection
    topk = sorted(enumerate(lens, 1), key=lambda x: x[1], reverse=True)[:3]
    print("\n가장 긴 청크 Top3 (id, tokens):", topk)

    return lens


def save_chunks_jsonl(chunks: List[Dict[str, int | str]], out_path: Path) -> None:
    """Write chunks to JSONL with sequential id and content hash cid."""
    with out_path.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks, 1):
            # Stable, content-based id (first 16 hex chars of SHA-256)
            obj = {
                "id": i,                         # sequential numeric id (1-based)
                "cid": cid16(ch["text"]),        # content hash id (stable across runs if text is identical)
                "tokens": ch["tokens"],          # token count for the chunk
                "text": ch["text"],              # raw chunk text
            }
            # One JSON object per line (JSONL format)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def preview_head_jsonl(path: Path, k: int = 3) -> None:
    """Print the first `k` records from a JSONL for a quick visual sanity check."""
    print("\n--- 미리보기 (앞 3개) ---")
    with path.open(encoding="utf-8") as f:
        for _ in range(k):
            line = f.readline()
            if not line:
                break  # Reached EOF before `k` lines
            o = json.loads(line)
            # Compact header line with id/cid/token count
            print(f"[{o['id']}] cid={o['cid']} · {o['tokens']} tokens")
            # Truncate preview to ~200 chars, collapse newlines
            print(o["text"][:200].replace("\n", " ") + "...\n")


def save_exact_dedup(in_path: Path, out_path: Path) -> Tuple[int, int]:
    """
    Create an exact-deduplicated JSONL by `cid` (keep first occurrence).
    Returns (original_count, kept_count).
    """
    seen, kept = set(), []
    total = 0

    # Scan the original JSONL and keep only the first occurrence of each cid
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            total += 1
            o = json.loads(line)
            if o["cid"] in seen:
                continue  # duplicate → skip
            seen.add(o["cid"])
            kept.append(o)

    # Write the deduplicated stream back out
    with out_path.open("w", encoding="utf-8") as f:
        for o in kept:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # Return counts for summary logging
    return total, len(kept)
```


```python
def main():
    # (A) Ensure tokenizer data
    ensure_punkt()

    # (B) Load file
    text = load_text_or_none(DOC_PATH)
    if text is None:
        return  # Early exit on missing file

    # (C) Clean + sentence split
    sents = preprocess_and_sentence_split(text)

    # (D) Build chunks
    chunks = build_chunks_from_sents(sents)

    # (E) Stats
    _ = summarize_chunks(chunks)

    # (F) Save JSONL with cid
    save_chunks_jsonl(chunks, OUT_JSONL)

    # (G) Preview first few
    preview_head_jsonl(OUT_JSONL, k=3)

    # (H) Save exact-deduped copy
    orig, kept = save_exact_dedup(OUT_JSONL, OUT_JSONL_DEDUP)
    print(f"Saved exact-deduplicated copy: {OUT_JSONL_DEDUP} (original {orig} → kept {kept})")


if __name__ == "__main__":
    main()
```

    === CHUNKS SUMMARY ===
    총 청크 수: 1378
    토큰수(평균/중앙/최소/최대): 85.74 / 89.0 / 18 / 294
    100 토큰 근접(90~100) 비율: 47.17%
    
    가장 긴 청크 Top3 (id, tokens): [(833, 294), (298, 236), (80, 178)]
    
    --- 미리보기 (앞 3개) ---
    [1] cid=978646441d4a7b5a · 79 tokens
    Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or...
    
    [2] cid=caf808fc8c97343d · 97 tokens
    firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amou...
    
    [3] cid=32e6d8adccd73ed5 · 55 tokens
    in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn’t think th...
    
    Saved exact-deduplicated copy: hp_chunks_100tok.dedup.jsonl (original 1378 → kept 1378)



```python
import re
from pathlib import Path

raw = Path("../../11_data/01 Harry Potter and the Sorcerers Stone.txt").read_text(encoding="utf-8", errors="ignore")
txt = clean_text(raw)

# 진짜 'M r.' (M과 r 사이에 공백류 1개 이상)
print("M r. 유형:", len(re.findall(r"(?i)m[\s\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000\uFEFF\u200B\u200C\u200D\u2060]+r\s*\.", txt)))

# Mr./Mrs./Ms./Dr./Prof. 뒤 공백 없음
print("Mr.뒤 공백 없음:", len(re.findall(r"\b(Mr|Mrs|Ms|Dr|Prof)\.(?=[A-Za-z])", txt)))

```

    M r. 유형: 0
    Mr.뒤 공백 없음: 0



```python

```


```python

```


```python

```


```python

```


```python

```
