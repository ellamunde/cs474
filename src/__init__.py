import preprocessing
import os
import txt_preprocessing

# txt = "Won the match #getin . Plus\u002c tomorrow is a very busy day\u002c with Awareness Day\u2019s and debates. Gulp. Debates..."
# txt_preprocessing.check_subjectivity(txt)
# txt_preprocessing.negation_detection(txt)
# txt_preprocessing.pos_tagger(txt)
# txt_preprocessing.tokenize(txt)
# print "----------------------------------"

directory = os.getcwd() + "/../data/train"
files = preprocessing.get_files(directory)
# print files
tweet_table = preprocessing.create_table(files)
tweet_table = preprocessing.add_preprocess_tweet(tweet_table)
# print tweet_table["TWEET"]
# preprocessing.get_all_tokens(tweet_table, "positive")
