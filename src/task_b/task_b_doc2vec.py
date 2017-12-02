# https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec
from pandas import DataFrame

from doc2vec import LabeledLineSentence
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import multiprocessing
import doc2vec
import preprocessing
import text_to_vector
import random
import numpy as np
# import statsmodels.api as sm


# test_b = preprocessing.get_data(test, "B")
dataset = 'B'
train_b = preprocessing.open_preprocess_file('train', dataset)

# --- get the lables, tweets, and polarities
topic_lables = train_b['TOPIC']
text = train_b['CLEANED']
polarity = train_b['POLARITY']
# print text

# --- get total of training instances and topics
num_train = len(topic_lables)
num_topics = len(topic_lables.value_counts())
print "total data"
print num_train
print "total polarity"
print polarity.value_counts()

# --- directory for model and dictionary
# dir_model = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/model/lda_", str(num_train), str(num_topics), str(passes),
#                                       str(alpha))
# dir_model = os.path.abspath(dir_model)
# dir_dict = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/dictionary/lda_", str(num_train), str(num_topics), str(passes),
#                                      str(alpha))
# dir_dict = os.path.abspath(dir_dict)

# --- preprocess
tokens_arr, sents_arr = preprocessing.preprocess(text)
# print type(sents_arr)
# sent_topic = topic_lables.to_frame().reset_index(drop=True).join(DataFrame({'TEXT': sents_arr}))
sent_topic = topic_lables.to_frame().reset_index(drop=True).join(DataFrame({'TEXT': sents_arr}))
# print sent_topic
topic_unique = list(set(topic_lables))
# topic_key = dict((topic_unique[i], i) for i in range(len(topic_unique)))
# topic_key = dict((topic_unique[i], i) for i in range(len(topic_unique)))

group = []
for topic in topic_unique:
    data = preprocessing.get_subset(table=sent_topic, topic=topic)
    # print data['TEXT']
    # for txt in data['TEXT']:
    #     print txt
    #     print row['TEXT']
    group.append([line for line in data['TEXT']])

print "group length: ", str(len(group))
print group

# print sents_arr[0]
sent_ids = ["sent_" + str(i) for i in range(len(sent_topic['TEXT']))]
# sentences = LabeledLineSentence(sents_arr, sent_ids)
# sent_ids = ["sent_" + str(i) for i in range(len(group))]
print sent_ids
# sentences = LabeledLineSentence(group, topic_unique)
# sentences = LabeledLineSentence(sent_topic['TEXT'], sent_topic['TOPIC'])
sentences = LabeledLineSentence(sent_topic['TEXT'], sent_ids)
# doc2vec_input = [sentences[id] for id in train_ids]
# sentences = word2vec.LineSentence([s.encode('utf-8').split() for s in sents_arr],polarity)

cores = multiprocessing.cpu_count()
print cores
model_DM = Doc2Vec(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=cores,  dm=1, dm_concat=1, batch_words=10000)
model_DBOW = Doc2Vec(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=cores, dm=0, batch_words=10000)

model_DM.build_vocab(sentences)
model_DBOW.build_vocab(sentences)

# print model_DM.docvecs.doctags
# print ">> vocabulary"
# print model_DM.docvecs

# for it in range(0,3):
#     random.shuffle(train_ids)
#     trainDoc = [doc2vec_input[id] for id in train_ids]
#     model_DM.train(doc2vec_input, total_examples=num_train, epochs=3)
#     model_DBOW.train(doc2vec_input, total_examples=num_train, epochs=3)

# start training
for epoch in range(200): #200
    if epoch % 20 == 0:
        print ('Now training epoch %s'%epoch)
    model_DM.train(sentences, total_examples=num_train, epochs=3)
    model_DM.alpha -= 0.002  # decrease the learning rate
    model_DM.min_alpha = model_DM.alpha  # fix the learning rate, no decay
    model_DBOW.train(sentences, total_examples=num_train, epochs=3)
    model_DBOW.alpha -= 0.002  # decrease the learning rate
    model_DBOW.min_alpha = model_DM.alpha  # fix the learning rate, no decay

random.seed(1)

#start testing
#printing the vector of document at index 1 in docLabels
# docvec = model_DM.docvecs[1]
print model_DM.docvecs[1]
print model_DBOW.docvecs[1]

#printing the vector of the file using its name
print model_DM.docvecs['sent_1'] #if string tag used in training
print model_DBOW.docvecs['sent_1'] #if string tag used in training
# print model_DM.docvecs[topic_unique[0]] #if string tag used in training
# print model_DBOW.docvecs[topic_unique[0]] #if string tag used in training

#to get most similar document with similarity scores using document-index
print model_DM.docvecs.most_similar(1)
print model_DBOW.docvecs.most_similar(1)

#to get most similar document with similarity scores using document- name
print model_DM.docvecs.most_similar('sent_1')
print model_DBOW.docvecs.most_similar('sent_1')
# print model_DM.docvecs.most_similar(topic_unique[0])
# print model_DBOW.docvecs.most_similar(topic_unique[0])
# print topic_unique[0]
#to get vector of document that are not present in corpus
# print model_DM.docvecs.infer_vector('x')
# print model_DBOW.docvecs.infer_vector('x')

# shows the similar words
# print (model_DM.most_similar('aaron'))
# print (model_DBOW.most_similar('aaron'))

# shows the learnt embedding
# print (model_DM['aaron'])
# print (model_DBOW['aaron'])

# shows the similar docs with id = 2
# print ">> vocabulary"
# print model_DM.docvecs
# print (model_DM.docvecs.most_similar(str(0)))
# print (model_DBOW.docvecs.most_similar(str(0)))

# you can save both word embeddings and document/paragraph embeddings:
# model.save('save/trained.model')
# model.save_word2vec_format('save/trained.word2vec')
# load the word2vec
# word2vec = gensim.models.Doc2Vec.load_word2vec_format('save/trained.word2vec')
# print (word2vec['good'])
# load the doc2vec
# model = gensim.models.Doc2Vec.load('save/trained.model')
# docvecs = model.docvecs


# print (docvecs[str(3)])

def plot_words(w2v):
    words_np = []
    # a list of labels (words)
    words_label = []
    for word in w2v.vocab.keys():
        words_np.append(w2v[word])
        words_label.append(word)
    print('Added %s words. Shape %s' % (len(words_np), np.shape(words_np)))

    pca = decomposition.PCA(n_components=2)
    pca.fit(words_np)
    reduced = pca.transform(words_np)

    # plt.plot(pca.explained_variance_ratio_)
    for index, vec in enumerate(reduced):
        # print ('%s %s'%(words_label[index],vec))
        if index < 100:
            x, y = vec[0], vec[1]
            plt.scatter(x, y)
            plt.annotate(words_label[index], xy=(x, y))
    plt.show()


# newindex = random.sample(range(0,num_train),num_train)
# testID = newindex[-TestNum:]
# trainID = newindex[:-TestNum]
# train_targets, train_regressors = zip(*[(Labels[id], list(model_DM.docvecs[id])+list(model_DBOW.docvecs[id])) for id in trainID])
# train_regressors = sm.add_constant(train_regressors)
# predictor = LogisticRegression(multi_class='multinomial',solver='lbfgs')
# predictor.fit(train_regressors,train_targets)

# --- init vectorizer
# vectorizer = text_to_vector.count_vectorizer()
# vectorizer = lda.tfidf_vectorizer()
# vectorizer = lda.fit_to_vectorizer(vectorizer, sents_arr)

# --- convert text into vectors using vectorizer
# bow_vectorizer = text_to_vector.fit_to_vectorizer(vectorizer, sents_arr)
# print ">> bow"
# print bow_vectorizer

# --- get feature names based on n-grams
# feature_names = text_to_vector.get_feature_names(vectorizer)

# --- convert dictionary to id2word
# idvec2word = text_to_vector.map_idvec2word(vectorizer)
# dict_len = len(idvec2word)

# --- convert bow vectorizer into bow lda
# bow_sparse = text_to_vector.convert_to_sparse_bow(bow_vectorizer)
# print ">> bow lda"

# train_b_sentences = doc2vec.get_sentences(train_b)
# word2vec_model = doc2vec.get_model(train_b_sentences)