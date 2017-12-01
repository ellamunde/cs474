import doc2vec
import preprocessing

train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"

train_b = preprocessing.get_data(train, "B")
# test_b = preprocessing.get_data(test, "B")

train_b_sentences = doc2vec.get_sentences(train_b)
word2vec_model = doc2vec.get_model(train_b_sentences)
