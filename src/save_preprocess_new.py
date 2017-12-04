import os

import preprocessing
from joblib import Parallel, delayed


def save_preprocess(text, type, task="A"):
    dirname = preprocessing.get_root_dir() + "/preprocess/sem/semeval_" + type + "_" + task + ".txt"
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
    print ">>>>>>>>>>>>>>>>>>> TEST A"
    print ">>>>>>>>>>>>>>>>>>> TEST A"
    print ">>>>>>>>>>>>>>>>>>> TEST A"
    train_a = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_test_A.txt")))

    save_preprocess(train_a, 'test', 'A')

    print ">>>>>>>>>>>>>>>>>>> TEST B"
    print ">>>>>>>>>>>>>>>>>>> TEST B"
    print ">>>>>>>>>>>>>>>>>>> TEST B"

    train_b = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_test_B.txt")))

    save_preprocess(train_b, 'test', 'B')

    print ">>>>>>>>>>>>>>>>>>> TEST C"
    print ">>>>>>>>>>>>>>>>>>> TEST C"
    print ">>>>>>>>>>>>>>>>>>> TEST C"
    train_c = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_test_C.txt")))

    save_preprocess(train_c, 'test', 'C')

    print ">>>>>>>>>>>>>>>>>>> TRAIN A"
    print ">>>>>>>>>>>>>>>>>>> TRAIN A"
    print ">>>>>>>>>>>>>>>>>>> TRAIN A"
    train_a = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_train_A.txt")))

    save_preprocess(train_a, 'train', 'A')

    print ">>>>>>>>>>>>>>>>>>> TRAIN B"
    print ">>>>>>>>>>>>>>>>>>> TRAIN B"
    print ">>>>>>>>>>>>>>>>>>> TRAIN B"
    train_b = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_train_B.txt")))

    save_preprocess(train_b, 'train', 'B')

    print ">>>>>>>>>>>>>>>>>>> TRAIN C"
    print ">>>>>>>>>>>>>>>>>>> TRAIN C"
    print ">>>>>>>>>>>>>>>>>>> TRAIN C"
    train_c = preprocessing.create_table(
        preprocessing.extract_txt(
            os.path.abspath("/Users/munde/PycharmProjects/cs474/semeval_train_2016/semeval_train_C.txt")))

    save_preprocess(train_c, 'train', 'C')


preprocess()