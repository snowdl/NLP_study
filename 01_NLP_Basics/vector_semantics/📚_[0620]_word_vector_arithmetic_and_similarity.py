# %%
import spacy

# %%
#loads spaCy's large English language model, en_core_web_lg
nlp  = spacy.load('en_core_web_lg')

# %%
#.shape reveals the dimensionality of the vector, typically (300,), meaning the word is represented in a 300-dimensional space.
nlp(u'lion').vector.shape

# %%
nlp(u'fox').vector.shape

# %%
tokens = nlp(u'like love hate')

# %%
'''
nested loop calculates the semantic similarity between all pairs of tokens in the tokens object (a spaCy Doc or Span).
token1.similarity(token2) returns a cosine similarity score based on their word vectors.
Values close to 1.0 mean high similarity, and values closer to 0 mean low or no similarity.
'''


for token1 in tokens :
    for token2 in tokens :
        print (token1.text, token2.text, token1.similarity(token2))

# %%
#checks how many word vectors are present
len(nlp.vocab.vectors)

# %%
nlp.vocab.vectors.shape

# %%
tokens = nlp(u"dog cat nurrgle hyeon")

# %%
#token.text – The word itself.
#token.has_vector – Whether this word has a word vector (True or False).
#token.vector_norm – The norm (magnitude) of the word vector. A value of 0.0 usually means no vector is present.
#token.is_oov – Whether the token is out of vocabulary (not included in the model's pretrained word vectors).

for token in tokens :
     print (token.text, token.has_vector, token.vector_norm, token.is_oov)

# %%
#step 1 : imports the spatial module from scipy.
#step 2 :  cosine_similarity ->omputes the cosine similarity between two vectors.
#1 - spatial.distance.cosine(vec1, vec2) converts cosine distance to cosine similarity.
#A result closer to 1 means higher similarity between the two vectors.
#step 3 :It loads the large English SpaCy model (en_core_web_lg) that includes pretrained word vectors.

# %%
from scipy import spatial
cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2)

nlp = spacy.load('en_core_web_lg')

# %%
king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

# %%
#king-man+women ----> new vector similar with queen, princess, queen, highness

# %%
new_vector = king-man+woman

# %%
computed_similarities = []
for word in nlp.vocab:
    if word.has_vector and word.is_lower and word.is_alpha:
        similarity = cosine_similarity(new_vector, word.vector)
        computed_similarities.append((word, similarity))  # 정확한 이름!


# %%
computed_similarities.sort(key=lambda x: -x[1])
print([t[0].text for t in computed_similarities[:10]])

# %%
print(len(computed_similarities))

# %%
# perform word vector arithmetic and find semantically similar words using SpaCy's pretrained vectors

# %%
#Step-by-step breakdown:
#Imports scipy.spatial and spacy.
#Loads the en_core_web_lg model, which includes pretrained word vectors.
#Defines a cosine similarity function using the cosine distance.
#Retrieves the vectors for "king", "man", and "woman".
#Computes a new vector via king - man + woman, which is a classic word analogy method (hoping for "queen").
#Iterates over the SpaCy vocabulary:
#Checks that the word has a vector, is lowercase, and contains only alphabetic characters.
#Calculates cosine similarity between each word's vector and the new vector.
Sorts the results by descending similarity.
Prints the top 10 most similar words — ideally, you'll see "queen" or similar concepts near the top.


# %%
from scipy import spatial
import spacy

nlp = spacy.load('en_core_web_lg')

cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

new_vector = king - man + woman

computed_similarities = []
for word in nlp.vocab:
    if word.has_vector and word.is_lower and word.is_alpha:
        similarity = cosine_similarity(new_vector, word.vector)
        computed_similarities.append((word, similarity))

computed_similarities.sort(key=lambda x: -x[1])

print([t[0].text for t in computed_similarities[:10]])


# %%
