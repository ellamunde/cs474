import pandas as pd
import os


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
        raw = raw + open('%s' % f, 'r').readlines()
    lines = [a.strip() for a in raw]
    return lines


def create_table(file_names):
    # files have duplicate rows
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


def preprocess(table,s_class):
    #get subset of table where polarity == to s_class
    table=table.loc[table['POLARITY']==s_class]
    tokens=[]
    for index,values in table.iterrows():
        tokens=tokens+extract_tokens(values['TWEET'])
    return tokens