# Bayes’ Theorem Notes

Bayes’ Theorem helps us update probabilities based on new evidence. It’s an important concept often used in NLP, medical diagnosis, spam filtering, and more, so I made sure to understand it well.

---

## 1. Bayes’ Theorem Formula

\[
P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}
\]

- \(P(A \mid B)\): Probability of A given B (what we want to find)
- \(P(B \mid A)\): Probability of B given A
- \(P(A)\): Prior probability of A
- \(P(B)\): Total probability of B

---

## 2. How to Calculate Total Probability \(P(B)\)

The denominator \(P(B)\) is calculated by:

\[
P(B) = P(B \mid A) \times P(A) + P(B \mid \neg A) \times P(\neg A)
\]

\(\neg A\) means “not A.”

---

## 3. Example I Studied: The Coin Problem

- 10% of coins are biased and show heads 90% of the time
- 90% of coins are fair and show heads 50% of the time
- We pick a coin at random and it lands heads up — what’s the probability that the coin is biased?

### 3-1. First, calculate total probability of heads \(P(\text{Heads})\):

\[
P(\text{Heads}) = P(\text{Heads} \mid \text{Biased}) \times P(\text{Biased}) + P(\text{Heads} \mid \text{Fair}) \times P(\text{Fair})
\]

\[
= 0.9 \times 0.1 + 0.5 \times 0.9 = 0.09 + 0.45 = 0.54
\]

---

### 3-2. Then use Bayes’ Theorem to find probability the coin is biased given heads:

\[
P(\text{Biased} \mid \text{Heads}) = \frac{P(\text{Heads} \mid \text{Biased}) \times P(\text{Biased})}{P(\text{Heads})} = \frac{0.9 \times 0.1}{0.54} \approx 0.167
\]

---

## 4. What I Learned

- Even if heads came up, the coin isn’t necessarily biased  
- Because biased coins are rare, the probability stays low  
- Bayes’ Theorem always takes the prior probability into account before updating beliefs

---

I found some parts confusing at first, but by calculating step-by-step, it became clearer.  
I think I can apply this approach to similar problems in the future!

---

Later, I plan to connect this to real NLP use cases like spam filtering.
