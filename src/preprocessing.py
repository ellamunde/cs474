import pandas as pd
import os
import nltk
import re
import io

from textblob import Word
from textblob import TextBlob


"""
get_files():
    Get all files in a directory
"""


def get_files(directory):
    abs_dir = os.path.abspath(directory)
    arr_file = []
    dirs = os.listdir(directory)

    for d in dirs:
        d = abs_dir + "/" + d
        print(d)
        arr_file.append(d)
        # testing
        # break

    return arr_file


"""
extract_txt():
    Reading text file per line
"""


def extract_txt(files):
    raw = []
    for f in files:
        # raw = raw + open(path+'%s'%f, 'r').readlines()
        raw = raw + io.open('%s' % f, 'r', encoding='utf-8').readlines()
    lines = [a.decode('utf-8').strip() for a in raw]
    return lines


def create_table(file_names):
    # files may have duplicated data
    txt = extract_txt(file_names)  # ['twitter-2013dev-A.txt','twitter-2013train-A.txt','twitter-2015train-A.txt'])
    table = pd.DataFrame()
    ids = []
    tweets = []
    polarity = []

    for t in txt:
        t = t.split('\t')
        # tweet id
        t_id = t[0]
        # tweet polarity
        t_pol = t[1]
        # tweet text
        t_text = t[2]
        ids.append(t_id)
        tweets.append(t_text)
        polarity.append(t_pol)

    table['ID'] = ids
    table['TWEET'] = tweets
    table['POLARITY'] = polarity

    # drop duplicates
    table.drop_duplicates('ID', inplace=True)
    return table


def preprocess_tweet(txt):
    txt = txt.lower()

    # normalize text
    txt = re.sub(r'\u2019', '\'', txt)
    print txt
    txt = txt.decode('unicode-escape')
    txt = txt.encode('utf-8')
    print txt

    txt = re.sub(r'n\'t', " not", txt)
    txt = re.sub(r'let\'s', "let us", txt)

    # convert hashtag into plain text
    # convert some of the characters into spaces
    # remove multiple spaces
    txt = re.sub(r'[#\\"]', " ", txt)
    txt = re.sub(r'\s+', " ", txt)
    print txt

    tokenizer = nltk.TweetTokenizer()
    tokens = tokenizer.tokenize(txt)

    new_tokens = []
    for token, tag in nltk.pos_tag(tokens):
        print tag + "-" + token

        # for link, replace them with special token
        # https://someweblog.com/url-regular-expression-javascript-link-shortener/
        if re.match(
                r"\(?(?:(http|https|ftp):\/\/)?(?:((?:[^\W\s]|\.|-|[:]{1})+)@{1})?((?:www.)?(?:[^\W\s]|\.|-)+[\.][^\W\s]{2,4}|localhost(?=\/)|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d*))?([\/]?[^\s\?]*[\/]{1})*(?:\/?([^\s\n\?\[\]\{\}\#]*(?:(?=\.)){1}|[^\s\n\?\[\]\{\}\.\#]*)?([\.]{1}[^\s\?\#]*)?)?(?:\?{1}([^\s\n\#\[\]]*))?([\#][^\s\n]*)?\)?",
                token):
            token = "%link"
        # ignore not-a-word token
        elif re.match(r"[^A-Za-z]+", tag):
            continue
        # ignore mention @
        elif re.match(r'^@', token):
            token = "%mention"
        # for number, replace them with special token
        # ignore not-a-word token
        elif tag == "CD":
            token = "%number"
        # for emoticon, replace them with special token
        # https://regex101.com/r/aM3cU7/4
        elif re.match(r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)", token):
            token = "%emoticon"
        # for words
        else:
            word = Word(token)
            print word.spellcheck()[0]
            spelling_token = word.spellcheck()[0][0]
            spelling_score = word.spellcheck()[0][1]

            if spelling_token != token and spelling_score == 1.0:
                if spelling_token.encode('utf-8') == token:
                    print "it is the same!"
                    continue

                token = spelling_token
                print "spelling score: " + "%3f" % spelling_score
                print "spelling token: " + token

        print token
        new_tokens.append(token.encode('utf-8'))

    print new_tokens
    return new_tokens


def add_preprocess_tweet(table):
    preprocess = []

    for index, row in table.iterrows():
        txt = " ".join(preprocess_tweet(row['TWEET']))
        preprocess += txt

    table['PREPROCESSED'] = preprocess
    return table


def get_all_tokens(table, polarity):
    # get subset table based on polarity type
    table = table.loc[table['POLARITY'] == polarity]
    tokens = []

    i = 0
    for index, row in table.iterrows():
        # tokens = tokens + preprocess_tweet(row['TWEET'])
        tokens = tokens + extract_tokens(row['TWEET'])
        # test only
        i += 1
        if i > 5:
            break

    return tokens


def extract_tokens(txt):
    txt = txt.lower()

    # import stopwords from nltk
    st_words = nltk.corpus.stopwords.words('english')

    # use tweet tokenizer from nltk
    tweet_tok = nltk.TweetTokenizer()
    tokens = tweet_tok.tokenize(txt)
    print tokens

    tblob = TextBlob(txt.decode('utf-8'))
    print tblob.detect_language()

    # remove stop words
    # tokens = [x for x in tokens if x not in st_words] # don't filter stopwords to extract negative words

    # attach pos tag extract only nouns, verbs, adjectives, adverbs
    tags = []

    # extend adjective pos tags
    tags.extend(['JJ', 'JJR', 'JJS'])

    # extend noun pos tags
    tags.extend(['NN', 'NNS', 'NNP', 'NNPS'])

    # extend adverb pos tags
    tags.extend(['RB', 'RBR', 'RBS'])

    # extend verb pos tags
    tags.extend(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    # print tags

    # tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    # tokens = [(token, tag) for token, tag in nltk.pos_tag(tokens) if tag in tags]
    # print tokens
    for token, tag in nltk.pos_tag(tokens):
        print tag + "-" + token

    return tokens

