import re

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as wnlemma


lemmatizer = wnlemma()


def lemmatize_words(tokens):
    new_tokens = []
    # print tokens
    for token, tag in tokens:
        s_not = ""
        if "_" in token:
            token = token.split("_")[0]
            s_not = "_not"

        new_lemma = get_lemma(token, tag).lower()
        new_tokens.append((new_lemma, tag))

    return tokens


def get_lemma(token, tag):
    pos = wn.NOUN

    if tag.startswith('J'):
        pos = wn.ADJ
    elif tag.startswith('V'):
        pos = wn.VERB
    elif tag.startswith('R'):
        pos = wn.ADV

    return lemmatizer.lemmatize(token, pos)


def add_synonyms(tokens):
    synonyms = []
    for token, tag in tokens:
        if re.match(r'^@', token):
            continue

        if "_" in token:
            token = token.split("_")[0]
            synonyms = synonyms + get_antonyms(token, tag)
        else:
            synonyms = synonyms + get_synonyms(token, tag)

    #append synonyms
    tokens = set(tokens + synonyms)
    return tokens


def add_antonyms(tokens, antonym_tokens):
    antonyms = []
    for token, tag in tokens:
        if re.match(r'^@', token):
            continue

        if "_" in token:
            token = token.split("_")[0]
            antonyms = antonyms + get_synonyms(token, tag)
        else:
            antonyms = antonyms + get_antonyms(token, tag)

    #append antonyms
    antonym_tokens = set(list(antonym_tokens) + antonyms)
    return antonym_tokens


def get_antonyms(token, tag):
    synsets = get_synsets(token, tag)

    antonyms = []
    for s in synsets:
        lemmas = s.lemmas()
        for l in lemmas:
            antonyms = antonyms + [(re.sub(r'[-_]+', ' ', w.name().lower()), tag) for w in l.antonyms()]
    return antonyms


def get_synonyms(token, tag):
    synsets = get_synsets(token, tag)

    synonyms = []
    for s in synsets:
        synonyms = synonyms + [(re.sub(r'[-_]+', ' ', w.name().lower()), tag) for w in s.lemmas()]
    return synonyms


def get_synsets(token, tag):
    if tag.startswith('N'):
        synsets = wn.synsets(token, 'n')
    elif tag.startswith('J'):
        synsets = wn.synsets(token, 'a')
    elif tag.startswith('R'):
        synsets = wn.synsets(token, 'r')
    else:
        synsets = wn.synsets(token, 'v')

    return synsets
