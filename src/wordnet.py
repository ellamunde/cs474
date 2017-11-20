import re

from nltk.corpus import wordnet as wn


def add_synonyms(tokens):
    synonyms = []
    for token, tag in tokens:
        if re.match(r'^@', token):
            continue

        synonyms = synonyms + get_synonyms(token, tag)

    #append synonyms
    tokens = set(tokens + synonyms)
    return tokens


def add_antonyms(tokens, antonym_tokens):
    antonyms = []
    for token, tag in tokens:
        if re.match(r'^@', token):
            continue
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
            antonyms = antonyms + [(w.name(), tag) for w in l.antonyms()]
    return antonyms


def get_synonyms(token, tag):
    synsets = get_synsets(token, tag)

    synonyms = []
    for s in synsets:
        synonyms = synonyms + [(w.name(), tag) for w in s.lemmas()]
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
