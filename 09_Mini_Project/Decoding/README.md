Speculative Decoding with N-gram + Prefix-Accept
📌 Overview

This mini-project demonstrates a beginner-friendly prototype of speculative decoding.

👉 The idea:

A small drafter proposes a short draft of tokens.

A larger verifier checks the draft and only accepts a verified prefix.

At the first mismatch, the verifier replaces the token and stops (prefix-accept rule).

We implement simple n-gram frequency models to simulate the drafter–verifier interaction.

⚙️ Components
1. Drafter (Unigram)

Samples k tokens based on unigram frequencies.

Includes an alpha parameter to sharpen (α>1) or flatten (α<1).

2. Verifiers

Bigram verifier → predicts P(next | prev1) (most frequent next token).

Backoff verifier (tri→bi→uni) → trigram if available, else bigram, else unigram.

3. Prefix-Accept Rule

Read draft tokens left→right.

Accept if drafter’s token == verifier’s token.

On first mismatch → replace with verifier’s token and stop.

🗂️ Implementation Notes

Language model = toy n-gram counts (no external LM).

Libraries: collections.Counter, defaultdict, random.

Designed for clarity, not performance.

🧪 Baseline vs Speculative

Prompt: ["the", "wolf", "ran"]
Draft length (k): 5
Drafter temperature: 0.9


📊 Backoff Demo
| Context `(prev2, prev1)` | What happens                                      | Distribution                                     | Sampled (T=1.0) | Argmax |
| ------------------------ | ------------------------------------------------- | ------------------------------------------------ | --------------- | ------ |
| `('wolf','ran')`         | Trigram found                                     | `{'into': 1}`                                    | always `into`   | `into` |
| `('ran','into')`         | Trigram found                                     | `{'the': 1}`                                     | always `the`    | `the`  |
| `('the','wolf')`         | Trigram found                                     | `{'ran': 1}`                                     | always `ran`    | `ran`  |
| `('hello','wolf')`       | Trigram missing → **backoff to bigram**           | `{'ran': 1}`                                     | always `ran`    | `ran`  |
| `('hello','zzz')`        | Trigram & bigram missing → **backoff to unigram** | `{'the':2,'wolf':1,'ran':1,'into':1,'forest':1}` | mostly `the`    | `the`  |


🔍 Key Takeaways

Out of 5 drafted tokens, on average 3.7 tokens are accepted before a mismatch.

This illustrates how speculative decoding combines efficiency (drafter) and accuracy (verifier).

Even with a toy corpus, the prefix-accept rule is clear and intuitive.
