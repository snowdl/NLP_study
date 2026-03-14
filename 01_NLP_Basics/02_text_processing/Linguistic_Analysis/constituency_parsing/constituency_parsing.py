import nltk
from nltk import Tree

sentence = "She enjoys reading scientific papers on natural language processing."

# Download required NLTK resources for grammar and parsing
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Tokenize the sentence into words
tokens = nltk.word_tokenize(sentence)

# Perform POS tagging on the tokenized words
pos_tags = nltk.pos_tag(tokens)

# Define a simple grammar for noun phrases (NP)
grammar = "NP: {<DT>?<JJ>*<NN>}"  # Regexp Grammar

# Create a RegexpParser with the defined grammar
cp = nltk.RegexpParser(grammar)

# Parse the POS-tagged sentence to identify noun phrases
result = cp.parse(pos_tags)

# Print the parsed tree in a readable format
result.pretty_print()

# Iterate over all subtrees labeled as 'NP' (Noun Phrase)
for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
    # Join and print the words forming the noun phrase
    print('Noun Phrase:', ' '.join(word for word, pos in subtree.leaves()))
