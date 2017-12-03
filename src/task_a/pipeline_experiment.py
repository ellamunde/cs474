import text_to_vector
from measurements import predict
import numpy
import lda
from sklearn.utils import shuffle
import pandas as pd
import pipeline
import preprocessing
import logistic_regression_multi as logres
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"
train_a = preprocessing.get_data(train, "A")
test_a = preprocessing.get_data(test, "A")
polarity_train=train_a['POLARITY']
polarity_test=test_a['POLARITY']
#pos_set=preprocessing.get_subset(train_a,'positive')
#neg_set=preprocessing.get_subset(train_a,'negative')
#neut_set=preprocessing.get_subset(train_a,'neutral')
#train_a=pd.concat([pos_set,neg_set,neut_set])
#train_a=shuffle(train_a)
# --- get the lables, tweets, and polarities
# --- state random
numpy.random.random(1)

# --- preprocess
text_train=train_a['CLEANED']
text_test=test_a['CLEANED']
train_model =pipeline.split_and_train(text_train,polarity_train,logres.initClassifier())
predict(text_test,polarity_test,train_model)

