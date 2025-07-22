import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("I love learning NLP")
for token in doc:
    print(f"{token.text:<10} -> {token.dep_:<10} (head: {token.head.text})")

doc = nlp("I study, write, and update my GitHub every night.")
for token in doc:
    print(f"{token.text:<10} → {token.dep_:<10} (head: {token.head.text})")

for token in doc:
    if token.dep_ == "nsubj":
        print("Subject:", token.text)
    elif token.dep_ in ("dobj", "pobj"):
        print("Object:", token.text)
    elif token.dep_ in ("ROOT", "conj") and token.pos_ == "VERB":
        print("Verb:", token.text)

doc = nlp("The book that John bought yesterday was expensive.")
for token in doc:
    print(f"{token.text:<10} → {token.dep_:<10} (head: {token.head.text})")

for token in doc:
    if token.dep_ == "nsubj":
        print("Subject:", token.text)
    elif token.dep_ == "dobj":
        print("Object:", token.text)
    elif token.dep_ in ("relcl", "ROOT"):
        print("Verb:", token.text)
    elif token.dep_ == "punct":
        print("Punct :", token.text)

triples = []
doc = nlp("Alice gave Bob a book and sent him an email.")
for token in doc:
    if token.dep_ == "nsubj":
        subject = token
        verb = token.head
        for child in verb.children:
            if child.dep_ in ("dobj", "pobj"):
                triples.append((subject.text, verb.text, child.text))

print(triples)
