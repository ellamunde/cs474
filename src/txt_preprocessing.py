import pattern.text.en as pattern
import nltk
import textblob.wordnet as wordnet

from textblob import TextBlob
from textblob import Word


def tokenize(text):
    nltk_tknz = nltk.word_tokenize(text)
    print nltk_tknz
    pttn_tknz = pattern.tokenize(text)
    print pttn_tknz
    txtb_tknz = TextBlob(text)
    print txtb_tknz.words
    print txtb_tknz.sentences


def check_subjectivity(text):
    process = TextBlob(text)
    subjectivity = process.sentiment.polarity
    print subjectivity
    return subjectivity


def pos_tagger(text):
    # print text
    nltk_tag = nltk.pos_tag(nltk.word_tokenize(text))
    print nltk_tag
    # pttn_tag = pattern.tag(text, relations=True, lemmata=True)
    # print pttn_tag
    txtb_tag = TextBlob(text).parse()
    print txtb_tag


def stopwords_removal(text):
    return text


def negation_detection(text):
    return text

