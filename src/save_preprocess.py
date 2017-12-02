import preprocessing
from joblib import Parallel, delayed


def save_preprocess(text, type="train", task="A"):
    dirname = preprocessing.get_root_dir() + "/preprocess/" + type + "_" + task + ".txt"
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
    # train_a = preprocessing.get_data('train', 'A')
    # save_preprocess(train_a, 'train', 'A')

    # train_b = preprocessing.get_data('train', 'B')
    # save_preprocess(train_b, 'train', 'B')
    #
    # train_c = preprocessing.get_data('train', 'C')
    # save_preprocess(train_c, 'train', 'C')

    test_a = preprocessing.get_data('test', 'A')
    save_preprocess(test_a, 'test', 'A')

    test_b = preprocessing.get_data('test', 'B')
    save_preprocess(test_b, 'test', 'B')

    test_c = preprocessing.get_data('test', 'C')
    save_preprocess(test_c, 'test', 'C')


preprocess()