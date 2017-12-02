import preprocessing

import features
import pandas as pd
import measurements
import logistic_regression_multi as logres
from sklearn.utils import shuffle
train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"


train_a = preprocessing.open_preprocess_file('train', 'A')
# tokens_arr, sents_arr = preprocessing.preprocess(text)
# train_a = preprocessing.get_data(train, "A")

pos_set=preprocessing.get_subset(train_a,'positive')[:500]
neg_set=preprocessing.get_subset(train_a,'negative')[:500]
neut_set=preprocessing.get_subset(train_a,'neutral')[:500]
train_a=pd.concat([pos_set,neg_set,neut_set])
train_a=shuffle(train_a)

# TODO: needs to replace this later...
test_a=preprocessing.open_preprocess_file('test', 'A')
tok_list=preprocessing.get_token_for_each_tweet(train_a)
tok_list_test=preprocessing.get_token_for_each_tweet(test_a)
#training
X=[]
Y=train_a['POLARITY']
for i in tok_list:
    tokens=i[0]
    word_count=i[1]
    polarity=i[2]
    
    vec=features.build_feature_vector(tokens,word_count)

    X.append(vec)
#clf=featureML.train_clf(X,Y)
print 'building model'
clf=logres.split_and_train(X, Y)
measurements.predict(X,Y,clf)

X_test=[]
Y_test=test_a['POLARITY']
for i in tok_list_test:
    tokens=i[0]
    word_count=i[1]
    polarity=i[2]

    vec=features.build_feature_vector(tokens,word_count)

    X_test.append(vec)

measurements.predict(X_test,Y_test,clf)