from utils import set_project_root
set_project_root()

import os
import spacy
from scipy import spatial


def download_spacy_model(model_name="en_core_web_lg"):
    """
    Check if spaCy model is installed; if not, download it.
    """
    import importlib.util

    if importlib.util.find_spec(model_name) is None:
        print(f"Downloading spaCy model '{model_name}'...")
        os.system(f"python -m spacy download {model_name}")
    else:
        print(f"spaCy model '{model_name}' is already installed.")


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    return 1 - spatial.distance.cosine(vec1, vec2)


def print_token_similarities(nlp, words):
    """
    Print pairwise semantic similarities between given tokens.
    """
    tokens = nlp(" ".join(words))
    for token1 in tokens:
        for token2 in tokens:
            print(f"{token1.text:<10} {token2.text:<10} {token1.similarity(token2):.4f}")


def analyze_tokens(nlp, words):
    """
    Print token properties: has_vector, vector_norm, is_oov.
    """
    tokens = nlp(" ".join(words))
    for token in tokens:
        print(
            f"{token.text:<10} has_vector: {token.has_vector:<5} "
            f"vector_norm: {token.vector_norm:.4f} is_oov: {token.is_oov}"
        )


def word_vector_arithmetic(nlp):
    """
    Perform king - man + woman vector arithmetic and find top 10 similar words.
    """
    king = nlp.vocab["king"].vector
    man = nlp.vocab["man"].vector
    woman = nlp.vocab["woman"].vector

    new_vector = king - man + woman

    computed_similarities = []
    for word in nlp.vocab:
        if word.has_vector and word.is_lower and word.is_alpha:
            similarity = cosine_similarity(new_vector, word.vector)
            computed_similarities.append((word.text, similarity))

    computed_similarities.sort(key=lambda x: -x[1])
    top_words = [word for word, sim in computed_similarities[:10]]
    print("Top 10 words similar to 'king - man + woman':", top_words)
    print(f"Total words with vectors considered: {len(computed_similarities)}")


def main():
    download_spacy_model()

    nlp = spacy.load("en_core_web_lg")

    # Vector shape check
    print("Shape of vector for 'lion':", nlp("lion").vector.shape)
    print("Shape of vector for 'fox':", nlp("fox").vector.shape)

    # Semantic similarity between tokens
    print("\nToken similarities:")
    print_token_similarities(nlp, ["like", "love", "hate"])

    # Number of word vectors and shape
    print("\nNumber of word vectors:", len(nlp.vocab.vectors))
    print("Shape of vocab vectors:", nlp.vocab.vectors.shape)

    # Token analysis
    print("\nToken properties:")
    analyze_tokens(nlp, ["dog", "cat", "nurrgle", "hyeon"])

    # Word vector arithmetic
    print("\nWord vector arithmetic results:")
    word_vector_arithmetic(nlp)


if __name__ == "__main__":
    main()
