# Speculative Decoding (n-gram mini project)

A minimal Draft & Verify demo using n-grams:
- **Drafter (small):** proposes *k* draft tokens using unigram/bigram frequencies
- **Verifier (large):** checks with **Trigram → Bigram → Unigram** backoff
- **Prefix-Accept:** immediately accept only the matching prefix; on first mismatch, replace with the verified token and stop

## Files
- `speculative_decoding_ngram_prefix_accept.ipynb`





