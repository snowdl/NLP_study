import sys
from collections import Counter

tokens = []
for line in sys.stdin:
    tokens += line.strip().split()

c = Counter(tokens)
print("total tokens:", sum(c.values()))
print("unique:", len(c))
print("top10:", c.most_common(10))
