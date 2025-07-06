from utils import set_project_root
set_project_root()

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources (only need to run once)
nltk.download('punkt')

def lemmatize_words(words, pos='n'):
    """Lemmatize a list of words with optional part of speech (default: noun)."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos=pos) for word in words]

def stem_words(words, stemmer):
    """Stem a list of words using the given stemmer instance."""
    return [stemmer.stem(word) for word in words]

def example_lemmatization():
    words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
    print('Before lemmatization:', words)
    print('After lemmatization (noun):', lemmatize_words(words, pos='n'))
    print('After lemmatization (verb):', lemmatize_words(words, pos='v'))

def example_stemming():
    words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()

    print('Before stemming:', words)
    print('PorterStemmer:', stem_words(words, porter_stemmer))
    print('LancasterStemmer:', stem_words(words, lancaster_stemmer))

def example_tokenize_and_stem():
    sentence = ("This was not the map we found in Billy Bones's chest, "
                "but an accurate copy, complete in all things--names and heights and soundings--"
                "with the single exception of the red crosses and the written notes.")
    tokenized_sentence = word_tokenize(sentence)
    porter_stemmer = PorterStemmer()

    print('Tokenized sentence:', tokenized_sentence)
    print('Stemmed sentence:', stem_words(tokenized_sentence, porter_stemmer))

if __name__ == "__main__":
    example_lemmatization()
    print()
    example_stemming()
    print()
    example_tokenize_and_stem()
