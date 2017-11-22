import pandas as pd
import nltk
import re
import io
import os
import enchant
import HTMLParser
# import string

from textblob import Word
from textblob import TextBlob


# import dictionary from enchant library
en_dictionary = enchant.Dict("en_US")
# import stopwords from nltk
st_words = nltk.corpus.stopwords.words('english')
# use tweet tokenizer from nltk
tweet_tok = nltk.TweetTokenizer()
# html parser
html_parser = HTMLParser.HTMLParser()


"""
get_data():
    Read data from text file and put it in data frame
"""


def get_data(data="train", task="A"):
    # directory for the text file
    directory = os.getcwd() + "/../data/" + task + "/" + data + "_" + task + ".txt"
    print directory
    directory = os.path.abspath(directory)
    # extract content of the file
    text = extract_txt(directory)
    # make data frame table
    table = create_table(text)
    return table


"""
extract_txt():
    Reading text file per line
"""


def extract_txt(filename):
    raw = io.open('%s' % filename, 'r', encoding='utf-8').readlines()
    lines = [a.decode('utf-8').strip() for a in raw]
    # print lines
    return lines


"""
create_table()
    create data frame for tweets data
"""


def create_table(txt):
    table = pd.DataFrame()
    tweets = []
    polarity = []
    topics = []
    cleaned = []

    for i in range(0, len(txt)):
        # DEVELOPMENT ONLY
        # if i > 50:
        #     break

        # split line
        t = txt[i].split('\t')
        # print t

        # tweet text
        t_text = t[0]
        t_cleaned = " ".join(preprocess_tweet(t_text))
        t_polarity = None

        # for task A
        if len(t) > 2:
            # tweet polarity
            t_topic = t[1]
            topics.append(t_topic)
            # tweet polarity
            t_polarity = t[2]
        # for task B and C
        else:
            # tweet polarity
            t_polarity = t[1]

        tweets.append(t_text)
        polarity.append(t_polarity)
        cleaned.append(t_cleaned)

    table['TWEET'] = tweets
    table['POLARITY'] = polarity
    table['CLEANED'] = cleaned

    if len(topics) > 0:
        table['TOPIC'] = topics

    # drop duplicates
    table.drop_duplicates('TWEET', inplace=True)
    # print len(txt)
    # print len(cleaned)
    # print len(tweets)
    # print len(table)
    return table


def preprocess_tweet(txt):
    txt = txt.lower()

    # normalize text
    txt = re.sub(r'\u2019', '\'', txt)
    print txt
    # txt = txt.decode('unicode-escape')
    # txt = txt.encode('utf-8')
    txt = html_parser.unescape(txt)
    print txt

    txt = re.sub(r'n\'t', "_not", txt)
    txt = re.sub(r'let\'s', "let us", txt)

    # convert hashtag into plain text
    # convert some of the characters into spaces
    # remove multiple spaces
    txt = re.sub(r'[#\\"]', " ", txt)
    txt = re.sub(r'\s+', " ", txt)
    # print txt

    tokenizer = nltk.TweetTokenizer()
    tokens = tokenizer.tokenize(txt)

    new_tokens = []
    for token, tag in nltk.pos_tag(tokens):
        # print tag + "-" + token

        # for link, replace them with special token
        # https://someweblog.com/url-regular-expression-javascript-link-shortener/
        if re.match(
                r"\(?(?:(http|https|ftp):\/\/)?(?:((?:[^\W\s]|\.|-|[:]{1})+)@{1})?((?:www.)?(?:[^\W\s]|\.|-)+[\.][^\W\s]{2,4}|localhost(?=\/)|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d*))?([\/]?[^\s\?]*[\/]{1})*(?:\/?([^\s\n\?\[\]\{\}\#]*(?:(?=\.)){1}|[^\s\n\?\[\]\{\}\.\#]*)?([\.]{1}[^\s\?\#]*)?)?(?:\?{1}([^\s\n\#\[\]]*))?([\#][^\s\n]*)?\)?",
                token):
            token = "@link"
        # ignore not-a-word token
        elif re.match(r"[^A-Za-z]+", tag):
            continue
        # ignore mention @
        elif re.match(r'^@', token):
            token = "@mention"
        # for number, replace them with special token
        # ignore not-a-word token
        elif tag == "CD":
            token = "@number"
        # for emoticon, replace them with special token
        # https://regex101.com/r/aM3cU7/4
        elif re.match(r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)", token):
            token = "@emoticon" + token
        # for words
        else:
            if not en_dictionary.check(token):
                word = Word(token)
                spelling_token = word.spellcheck()[0][0]
                spelling_score = word.spellcheck()[0][1]
                # print "spelling score: " + "%3f" % spelling_score
                # print "spelling token: " + spelling_token
                # print "token: " + token

                if spelling_token != token and spelling_score == 1.0:
                    if spelling_token.encode('utf-8') == token:
                        # print "it is the same!"
                        continue

                    token = spelling_token
                    # print "spelling score: " + "%3f" % spelling_score
                    # print "spelling token: " + token

        # print token
        new_tokens.append(token.encode('utf-8'))

    # print new_tokens
    return new_tokens


def get_tokens(table, polarity=None, data='CLEANED'):
    # get subset table based on polarity type
    subset = get_subset(table, polarity)
    # print subset
    tokens = []

    for index, row in subset.iterrows():
        tokens = tokens + extract_tokens(row[data])

    return tokens


def get_subset(table, polarity=None):
    subset = None
    if polarity is not None:
        subset = table.loc[table['POLARITY'] == polarity]
    else:
        return table

    return subset


def extract_tokens(txt):
    txt = txt.lower()

    tokens = tweet_tok.tokenize(txt)
    print tokens

    tags = get_tags()
    tokens = [(token, tag) for token, tag in nltk.pos_tag(tokens) if tag in tags]

    return tokens


def detect_lang(txt):
    tblob = TextBlob(txt.decode('utf-8'))
    lang = tblob.detect_language()
    print lang
    return lang


def remove_stopwords(tokenize_txt):
    # remove stop words
    tokens = [x for x in tokenize_txt if x not in st_words] # don't filter stopwords to extract negative words
    return tokens


def get_tags():
    tags = []

    # extend adjective pos tags
    tags.extend(['JJ', 'JJR', 'JJS'])

    # extend noun pos tags
    tags.extend(['NN', 'NNS', 'NNP', 'NNPS'])

    # extend adverb pos tags
    tags.extend(['RB', 'RBR', 'RBS'])

    # extend verb pos tags
    tags.extend(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    return tags


def get_tokens_only(tokens):
    new_tokens = [token for token, tag in tokens]
    return new_tokens


"""
filter tokens:
leave only those that appear in one group..later use pmi or idf to filter
"""


def filter_tokens(tokens, tokens_to_remove):
    uniq_tokens = set(tokens) - set(tokens_to_remove)
    tokens = [t for t in tokens if t in uniq_tokens]
    return tokens
