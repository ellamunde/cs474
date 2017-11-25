from __future__ import division
import preprocessing

import featureML

train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"
train_a = preprocessing.get_data(train, "A")
test_a=preprocessing.get_data(test, "A")
#equal number of instances for each class
pos_set=preprocessing.get_subset(train_a,'positive')
neg_set=preprocessing.get_subset(train_a,'negative')
neut_set=preprocessing.get_subset(train_a,'neutral')

#takes too much time to process limit to 500 instances only
tokens_pos = preprocessing.get_tokens(pos_set[:500])
tokens_neg = preprocessing.get_tokens(neg_set[:500])
tokens_neu = preprocessing.get_tokens(neut_set[:500])

tokens_pos=preprocessing.get_tokens_only(tokens_pos)
tokens_neg=preprocessing.get_tokens_only(tokens_neg)
tokens_neu=preprocessing.get_tokens_only(tokens_neu)

print len(tokens_pos), len(tokens_neg),len(tokens_neu)

features=preprocessing.filter_pmi([tokens_pos,tokens_neg,tokens_neu])
print 'creating feature vectors for training'
X,Y=featureML.create_X_Y(train_a[:2000],features)

clf=featureML.train_clf(X,Y)

pred_result=clf.predict(X)

print 'train prediction results'
featureML.get_accuracy(pred_result,Y)

print 'creating feature vectors for testing'
X_test,Y_test=featureML.create_X_Y(test_a,features)
pred_test=clf.predict(X_test)

print 'test prediction results'
featureML.get_accuracy(pred_test,Y_test)
