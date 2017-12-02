from __future__ import division
import pandas as pd
import nltk
import re
import io
import os
import enchant
import HTMLParser
import numpy as np
import wordnet

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from textblob import Word

##########
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
##########

# import dictionary from enchant library
en_dictionary = enchant.Dict("en_US")
# import stopwords from nltk
st_words = nltk.corpus.stopwords.words('english')
# use tweet tokenizer from nltk
tweet_tok = nltk.TweetTokenizer()
# html parser
html_parser = HTMLParser.HTMLParser()


def get_root_dir():
    """
    get_data():
    - Read data from text file and put it in data frame
    """
    getcwd = os.getcwd
    directory = getcwd()

    basename = os.path.basename
    listdir = os.listdir

    while True:
        # path = basename(getcwd())
        folders = [basename(f) for f in listdir(getcwd())]
        if 'src' in folders:
            return os.path.abspath(directory)

        os.chdir("..")
        directory += "/.."


def get_data(data="train", task="A"):
    # directory for the text file
    rootdir = get_root_dir()
    # directory = os.getcwd() + "/data/" + task + "/" + data + "_" + task + ".txt"
    # directory = "s%s%/s%/s%_s%s%".format(getcwd(), "/data", task, data, task, ".txt")
    directory = "{}{}/{}/{}_{}{}".format(rootdir, "/data", task, data, task, ".txt")

    # extract content of the file
    # make data frame table
    return create_table(extract_txt(os.path.abspath(directory)))


def open_preprocess_file(data="train", task="A"):
    rootdir = get_root_dir()
    directory = "{}{}/{}_{}{}".format(rootdir, "/preprocess", data, task, ".txt")

    table = pd.DataFrame()
    tweets = []
    polarity = []
    topics = []
    cleaned = []
    cleaned_token = []

    appendtweets = tweets.append
    appendpolarity = polarity.append
    appendtopics = topics.append
    appendcleaned = cleaned.append
    appendcleanedtoken = cleaned_token.append
    join = " ".join

    for line in tqdm(extract_txt(os.path.abspath(directory))):
        # f.write(row['TWEET'])
        # f.write("\t")
        # f.write(row['CLEANED'])
        # f.write("\t")
        # f.write(",".join(row['CLEANED_TOKEN']))
        # f.write("\t")
        # if task != "A":
        #     f.write(row['TOPIC'])
        #     f.write("\t")
        # f.write(row['POLARITY'])
        # f.write("\n")

        line = line.split('\t')
        # tweet text
        t_tweet = line[0]
        t_cleaned = line[1]
        t_cleaned_tokens = line[2]

        if len(line) > 4:
            # tweet polarity
            appendtopics(line[3])
            # tweet polarity
            t_polarity = line[4]
        # for task B and C
        else:
            # tweet polarity
            t_polarity = line[3]


        appendtweets(t_tweet)
        appendpolarity(t_polarity)
        appendcleaned(t_cleaned)
        appendcleanedtoken(t_cleaned_tokens)

    table['TWEET'] = tweets
    table['POLARITY'] = polarity
    table['CLEANED'] = cleaned
    table['CLEANED_TOKEN'] = cleaned_token

    if len(topics) > 0:
        table['TOPIC'] = topics

    # drop duplicates
    table.drop_duplicates('TWEET', inplace=True)
    return table


def extract_txt(filename):
    """
    extract_txt():
    - Reading text file per line
    """
    # raw = io.open('%s' % filename, 'r', encoding='utf-8').readlines()
    lines = [a.decode('utf-8').strip() for a in io.open('%s' % filename, 'r', encoding='utf-8').readlines()]
    # print lines
    return lines


def create_table(txt):
    """
    create_table()
    create data frame for tweets data
    """
    table = pd.DataFrame()
    tweets = []
    polarity = []
    topics = []
    cleaned = []
    cleaned_token = []

    appendtweets = tweets.append
    appendpolarity = polarity.append
    appendtopics = topics.append
    appendcleaned = cleaned.append
    appendcleanedtoken = cleaned_token.append
    join = " ".join

    for i in tqdm(range(0, len(txt))):
        # DEVELOPMENT ONLY
        # if i > 30:
        #     break

        # split line
        t = txt[i].split('\t')
        # print t

        # tweet text
        t_text = t[0]
        cleaned_text = preprocess_tweet(t_text)
        t_cleaned = join(cleaned_text)
        # t_polarity = None

        # for task B, C
        if len(t) > 2:
            # tweet polarity
            appendtopics(t[1])
            # tweet polarity
            t_polarity = t[2]
        # for task B and C
        else:
            # tweet polarity
            t_polarity = t[1]

        appendtweets(t_text)
        appendpolarity(t_polarity)
        appendcleaned(t_cleaned)
        appendcleanedtoken(cleaned_text)

    table['TWEET'] = tweets
    table['POLARITY'] = polarity
    table['CLEANED'] = cleaned
    table['CLEANED_TOKEN'] = cleaned_token

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
    new_tokens = []
    txt = txt.lower()

    replace = re.sub
    ismatch = re.match
    appendnewtoken = new_tokens.append

    # normalize text
    txt = replace(r'\u2019', '\'', txt)
    print txt
    # txt = txt.decode('unicode-escape')
    # txt = txt.encode('utf-8')
    # txt = html_parser.unescape(txt)
    # print txt

    txt = html_parser.unescape(txt)

    txt = replace(r'n\'t', "_not", txt)
    txt = replace(r'let\'s', "let us", txt)
    txt = replace(r'[#\\"]', " ", txt)
    txt = replace(r'\s+', " ", txt)

    # convert hashtag into plain text
    # convert some of the characters into spaces
    # remove multiple spaces
    # txt = replace(r'[#\\"]', " ", txt)
    # txt = replace(r'\s+', " ", txt)
    # print txt

    tokenizer = nltk.TweetTokenizer()
    tokens = tokenizer.tokenize(txt)

    for token, tag in nltk.pos_tag(tokens):
        # print tag + "-" + token

        # for link, replace them with special token
        # https://someweblog.com/url-regular-expression-javascript-link-shortener/
        if ismatch(
                r"\(?(?:(http|https|ftp):\/\/)?(?:((?:[^\W\s]|\.|-|[:]{1})+)@{1})?((?:www.)?(?:[^\W\s]|\.|-)+[\.][^\W\s]{2,4}|localhost(?=\/)|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d*))?([\/]?[^\s\?]*[\/]{1})*(?:\/?([^\s\n\?\[\]\{\}\#]*(?:(?=\.)){1}|[^\s\n\?\[\]\{\}\.\#]*)?([\.]{1}[^\s\?\#]*)?)?(?:\?{1}([^\s\n\#\[\]]*))?([\#][^\s\n]*)?\)?",
                token):
            token = "@link"
        # ignore not-a-word token
        elif ismatch(r"[^A-Za-z]+", tag):
            continue
        # ignore mention @
        # elif re.match(r'^@', token):
        #     token = "@mention"
        # for number, replace them with special token
        # ignore not-a-word token
        elif tag == "CD":
            token = "@number"
        # for emoticon, replace them with special token
        # https://regex101.com/r/aM3cU7/4
        elif ismatch(r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)", token):
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
        appendnewtoken(token.encode('utf-8'))

    # print new_tokens
    return new_tokens


def get_tokens(table, polarity=None, data='CLEANED'):
    # get subset table based on polarity type
    if polarity is not None:
        subset = get_subset(table, polarity)
    else:
        subset = table
    # print subset
    tokens = [extract_tokens(row[data]) for index, row in subset.iterrows()]
    # tokens = []
    # print ">> extractions:"
    # for index, row in subset.iterrows():
        # extractions = extract_tokens(row[data])
        # tokens += extract_tokens(row[data])
        # print extractions
        # tokens += extractions

    return tokens


def get_subset(table, polarity=None, topic=None):
    subset = table
    if polarity is not None:
        subset = table.loc[table['POLARITY'] == polarity]

    if topic is not None:
        subset = subset.loc[table['TOPIC'] == topic]

    return subset


def extract_tokens(txt):
    tokens = tweet_tok.tokenize(txt.lower())
    # print tokens
    tokens = [(token, tag) for token, tag in nltk.pos_tag(tokens) if tag in get_tags()]
    return tokens


# def detect_lang(txt):
#     tblob = TextBlob(txt.decode('utf-8'))
#     lang = tblob.detect_language()
#     print lang
#     return lang


def remove_stopwords(tokenize_txt):
    # remove stop words
    # print st_words
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


def filter_tokens(tokens, tokens_to_remove):
    """
    filter tokens:
    leave only those that appear in one group..later use pmi or idf to filter
    """
    unique_tokens = set(tokens) - set(tokens_to_remove)
    tokens = [t for t in tokens if t in unique_tokens]
    return tokens


def filter_pmi(sets,th=0.9):
    """
    returns only tokens with pmi higher than threshold
    sets = positive negative neutral
    """
    result=[]
    for i in range(0,len(sets)):
        pmi_set={}
        #polarity class
        cl=sets[i]
        indices=[idx for idx in range(0,len(sets)) if idx!=i]
        print indices
        other=[]
        all_tokens=cl
        for idx in indices:
            other.append(sets[idx])
            all_tokens.extend(sets[idx])
        tot_num_cl=len(cl)
        tot_num=sum([len(a) for a in sets])
        for token in set(cl):
            pr_cl_token=cl.count(token)/tot_num_cl
            pr_cl=tot_num_cl/tot_num
            pr_token=all_tokens.count(token)/tot_num_cl
            pmi=np.log2(pr_cl_token/(pr_cl*pr_token))
            print pmi
            if pmi>=th:
                pmi_set[token]=pmi

        result.extend(pmi_set.keys())
        print result
    return result


def split_data(text, label, test_size=0.2, random_state=8):
    text_train, text_test, label_train, label_test = train_test_split(
        text, label, test_size=test_size, random_state=random_state
    )

    return text_train, text_test, label_train, label_test


def preprocess(text):
    join = " ".join
    tokens = [extract_tokens(row) for row in tqdm(text)]
    lemma = [wordnet.lemmatize_words(x) for x in tqdm(tokens)]

    no_stopwords = []
    no_stopwords_sent = []
    for x in tqdm(lemma):
        # print x
        x = remove_stopwords(
            get_tokens_only(x)
        )
        no_stopwords.append(x)
        no_stopwords_sent.append(join(x))
        # print x

    return no_stopwords, no_stopwords_sent




####################################method with no bag of words#####################################
def get_lemmas(token,tag):

    token=wordnet_lemmatizer.lemmatize(token,tag)

    return (token,tag)

def get_words(txt):
    words = []
    for line in txt:
        line = line.strip()
        if line != '' and not line.startswith(';'):
            words.append(line)
    return words
 #######################lexicon#####################will standartize path later#####


def add_polarity(tokens):
    negative = open(get_root_dir() + '/src/task_a/negative-words.txt').readlines()
    positive = open(get_root_dir() + '/src/task_a/positive-words.txt').readlines()

    negative = get_words(negative)
    positive = get_words(positive)
    result=[]
    for token,tag in tokens:
        dict={'pos':0,'neg':0,'neu':0}
        try:
            pol = swn.senti_synset('%s.%s.01' % (token, tag))
            dict['pos']=pol.pos_score()
            dict['neg']=pol.neg_score()
            dict['neu']=pol.obj_score()
            polarity=max(dict.keys(), key=(lambda k:dict[k]))
        except:
            #if not found check in lexicon
            if token in negative:
                polarity='neg'

            elif token in positive:
                polarity='pos'
            elif token.startswith("%"):
                polarity='neu'
            else:
                #may be not include tokens with not recognized polarity?
                continue
        result.append((token,tag,polarity))
    return result


def get_token_for_each_tweet(table):
    tags = get_tags()
    token_list=[]
    pattern = r'%?[a-z]+'
    tokenizer = nltk.RegexpTokenizer(pattern)
    for i,values in table.iterrows():
        txt=values['CLEANED']
        p=values['POLARITY']
        tokens = tokenizer.tokenize(txt)
        tokens = [(token, tag) for token, tag in nltk.pos_tag(tokens) if tag in tags]
        num_words=len(tokens)
        new_tokens=[]
        #change tag format to use with lemmatizer and senti wordnet
        for token,tag in tokens:

            if tag.startswith('J'):
                new_tokens.append((token,'a'))
            elif tag.startswith('N'):
                t,tg=get_lemmas(token,'n')
                new_tokens.append((t,tg))
            elif tag.startswith('R'):
                new_tokens.append((token,'r'))
            elif tag.startswith('V'):
                t,tg=get_lemmas(token,'v')
                new_tokens.append((t, tg))

        #tokens=set(get_tokens_only(tokens))
        new_tokens=add_polarity(new_tokens)
        if len(new_tokens)==0:
            continue
        token_list.append((new_tokens,num_words,p))

    return token_list
#####################################################################################################################################
