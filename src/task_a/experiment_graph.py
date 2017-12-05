import src.preprocessing as preprocessing
import src.graph as graph
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
import numpy as np

train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"
train_a = preprocessing.open_preprocess_file(train, "A")
test_a=preprocessing.open_preprocess_file(test, "A")

#build class graphs on these sets
pos_set=preprocessing.get_subset(train_a,'positive')[:500]
neg_set=preprocessing.get_subset(train_a,'negative')[:500]
neut_set=preprocessing.get_subset(train_a,'neutral')[:500]
print len(pos_set), len(neg_set),len(neut_set)

#train classifier on these  sets
pos_train=preprocessing.get_subset(train_a,'positive')[500:]
neg_train=preprocessing.get_subset(train_a,'negative')[500:]
neut_train=preprocessing.get_subset(train_a,'neutral')[500:]
print len(pos_train), len(neg_train),len(neut_train)

#tokenization need tokens as list for each tweet without pos
def get_tokens(subset):
    tokens = []

    for index, row in subset.iterrows():
        t=preprocessing.tweet_tok.tokenize(row['CLEANED'])
        tokens.append(t)

    print tokens
    return tokens

pos_graph=graph.build_class_graph(get_tokens(pos_set))
neg_graph=graph.build_class_graph(get_tokens(neg_set))
neut_graph=graph.build_class_graph(get_tokens(neut_set))


graph_col=[pos_graph, neg_graph,neut_graph]
full_train = (pos_train.append(neg_train)).append(neut_train)
full_train = shuffle(full_train)

full_train['tokens']=get_tokens(full_train)

#X=feature vectors for classifier
#Y=target value
def create_X_Y(data):
    X = []
    Y = []
    for i,vals in data.iterrows():
        tokens=vals['tokens']
        tweet_graph=graph.create_graph(tokens)
        feature_vec=[]
        for class_graph in graph_col:
            cs=graph.calc_cs(tweet_graph,class_graph)
            feature_vec.append(cs)
            vs=graph.calc_vs(tweet_graph,class_graph)
            feature_vec.append(vs)
            nvs=graph.calc_nvs(tweet_graph,class_graph,vs)
            feature_vec.append(nvs)
        print feature_vec
        X.append(feature_vec)
        Y.append(vals['POLARITY'])
    return X,Y
X,Y=create_X_Y(full_train)
X_test,Y_test=create_X_Y(test_a)

X=np.array(X)
Y=np.array(Y)
#X_test=np.array(X_test)
clf = MultinomialNB()
clf.fit(X, Y)
pred_result=clf.predict(X)
