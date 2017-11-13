import preprocessing
import os


directory = os.getcwd() + "/../data/train"
files = preprocessing.get_files(directory)
# print files
tweet_table = preprocessing.create_table(files)
print tweet_table["TWEET"]
