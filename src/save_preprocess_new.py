import os

import preprocessing
from joblib import Parallel, delayed


def save_preprocess(text, type, task="A"):
    dirname = preprocessing.get_root_dir() + "/preprocess/semeval_" + task + ".txt"
    f = open(dirname, 'w')
    for idx, row in text.iterrows():
        f.write(row['TWEET'])
        f.write("\t")
        f.write(row['CLEANED'])
        f.write("\t")
        f.write(",".join(row['CLEANED_TOKEN']))
        f.write("\t")
        if task != "A":
            f.write(row['TOPIC'])
            f.write("\t")
        f.write(row['POLARITY'])
        f.write("\n")
        # break

    f.close()


def preprocess():
    train_a = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_test_A.txt")))

    save_preprocess(train_a, 'test', 'A')

    train_b = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_test_B.txt")))

    save_preprocess(train_b, 'test', 'B')

    train_c = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_test_C.txt")))

    save_preprocess(train_c, 'test', 'C')


preprocess()