# 📘 Summary of Today's Study: Probability Functions (Focus on PMF)

## 1️⃣ What is a Probability Function?

A **probability function** describes how likely a random variable is to take on a specific value.

There are two main types:
- **PMF (Probability Mass Function)**: for **discrete** random variables
- **PDF (Probability Density Function)**: for **continuous** random variables

---

## 2️⃣ PMF: Probability Mass Function

- Defines the probability that a **discrete random variable** takes on a specific value
- Must satisfy:
  \[
  0 \leq P(X = x) \leq 1
  \]
  \[
  \sum P(X = x) = 1
  \]

---

## 3️⃣ Example 1: Tossing Two Coins

Let the random variable **X = number of heads**.  
Possible outcomes: 0, 1, 2

| X (heads) | P(X = x) |
|-----------|----------|
| 0         | 1/4      |
| 1         | 2/4      |
| 2         | 1/4      |

Total probability:  
\[
P(X = 0) + P(X = 1) + P(X = 2) = 1
\]

---

## 4️⃣ Example 2: Rolling Two Dice

Let the random variable **X = sum of the two dice**  
Possible values: 2 to 12  
Total outcomes: 6 × 6 = 36

| X | # of combinations | P(X = x)  |
|---|-------------------|-----------|
| 2 | 1                 | 1/36      |
| 3 | 2                 | 2/36      |
| 4 | 3                 | 3/36      |
| 5 | 4                 | 4/36      |
| 6 | 5                 | 5/36      |
| 7 | 6                 | 6/36      |
| 8 | 5                 | 5/36      |
| 9 | 4                 | 4/36      |
|10 | 3                 | 3/36      |
|11 | 2                 | 2/36      |
|12 | 1                 | 1/36      |

---

## 5️⃣ Shortcut Rule for Dice PMF

Instead of listing all combinations, you can use this formula:

\[
\text{# of combinations} =
\begin{cases}
X - 1, & \text{if } 2 \leq X \leq 7 \\
13 - X, & \text{if } 8 \leq X \leq 12
\end{cases}
\]

Example:  
For X = 9 → combinations = 13 - 9 = 4 → P(X = 9) = 4/36

---

## 💬 What You Did Well Today

- Correctly identified total number of outcomes (36)
- Attempted to count the combinations manually — great initiative!
- Asked insightful questions like “Why does the probability not sum to 3/4?”
- Learned and applied the shortcut rule for PMF of dice rolls

---


