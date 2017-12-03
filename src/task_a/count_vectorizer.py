import text_to_vector
from measurements import predict
import numpy
import lda
from sklearn.utils import shuffle
import pandas as pd
import preprocessing
import logistic_regression_multi as logres
train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"
train_a = preprocessing.open_preprocess_file(train, "A")
test_a = preprocessing.open_preprocess_file(test, "A")
#pos_set=preprocessing.get_subset(train_a,'positive')
#neg_set=preprocessing.get_subset(train_a,'negative')
#neut_set=preprocessing.get_subset(train_a,'neutral')
#train_a=pd.concat([pos_set,neg_set,neut_set])
#train_a=shuffle(train_a)
# --- get the lables, tweets, and polarities
text = train_a['CLEANED']
polarity = train_a['POLARITY']
test=test_a['CLEANED']
polarity_t=test_a['POLARITY']
# --- state random
numpy.random.random(1)

# --- preprocess
tokens_arr, sents_arr = preprocessing.preprocess(text)

# --- init vectorizer
vectorizer = text_to_vector.count_vectorizer()
print vectorizer
# vectorizer = lda.tfidf_vectorizer()
# vectorizer = lda.fit_to_vectorizer(vectorizer, sents_arr)

# --- convert text into vectors using vectorizer
bow_vectorizer = text_to_vector.fit_to_vectorizer(vectorizer, sents_arr)
train_model =logres.split_and_train(bow_vectorizer, polarity)
tokens_arr, sents_arr = preprocessing.preprocess(test)

# --- init vectorizer

# vectorizer = lda.tfidf_vectorizer()
# vectorizer = lda.fit_to_vectorizer(vectorizer, sents_arr)

# --- convert text into vectors using vectorizer
bow_vectorizer = text_to_vector.transform_text(vectorizer, sents_arr)
prediction_res = predict(bow_vectorizer, polarity_t, train_model)