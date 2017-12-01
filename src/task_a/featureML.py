from __future__ import division
import preprocessing
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pickle

#s=open('features.pkl','rb')
#features=pickle.load(s)
#s.close()

#create X (feature vectors) Y (Target value)
#fs=features
def create_X_Y(data,fs):
    X = []
    Y = []
    for i,vals in data.iterrows():
        t = preprocessing.tweet_tok.tokenize(vals['CLEANED'])

        feature_vec=[]
        #so far vectores created based on presence/absence of feature
        for v in fs:
            if v in t:
                feature_vec.append(1)
            else:
                feature_vec.append(0)
            #print feature_vec
        X.append(feature_vec)

        Y.append(vals['POLARITY'])

    return np.array(X),np.array(Y)



def train_clf(X,Y):
    print 'training'
    clf = MultinomialNB()
    clf.fit(X, Y)
    return clf


def get_accuracy(predicted,actual):
    pos_num=list(actual).count('positive')
    neg_num = list(actual).count('negative')
    neut_num = list(actual).count('neutral')
    correctly_class=0
    pos=0
    neg=0
    neut=0
    for i in range(0,len(predicted)):
        if predicted[i]==actual[i]:
            correctly_class+=1
            if predicted[i]=='positive':
                pos+=1
            elif predicted[i]=='negative':
                neg+=1
            else:
                neut+=1
    print 'number of instance %d: positive %d, negative %d, neutral %d' %(len(actual),pos_num,neg_num,neut_num)
    print 'overall accuracy'
    print correctly_class / len(actual)
    print 'positive: '
    print pos/pos_num
    print 'negative'
    print neg/neg_num
    print 'neutral'
    print neut/neut_num









