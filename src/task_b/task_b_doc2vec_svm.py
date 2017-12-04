import pre_task_bc_doc2vec as pre

dataset = 'B'
input_from_file = pre.get_data('train', dataset)[:100]
test_set = pre.get_data('test', dataset)[:100]

epoch = 3
model_dm, model_dbow, train_data = pre.get_model(input_from_file, epoch=epoch)

dm_svm_model = pre.polarity_model(d2v_model=model_dm,model='svm', train_data=train_data, multi=False)
dbow_svm_model = pre.polarity_model(d2v_model=model_dbow,model='svm', train_data=train_data, multi=False)

print ">> dm, svm-dm"
dm_prediction = pre.polarity_test(model_dm, dm_svm_model, test_set)
print ">> dbow, svm-dbow"
dbow_prediction = pre.polarity_test(model_dbow, dbow_svm_model, test_set)


import measurements as m
m.get_accuracy(dm_prediction)
m.avg_recall(dm_prediction)
m.get_accuracy(dbow_prediction)
m.avg_recall(dbow_prediction)


# print ">> dbow, svm-dm"
# dbow_dm_prediction = pre.svm_polarity_test(model_dbow, dm_svm_model, dataset='B')
# print ">> dm, svm-dbow"
# dm_dbow_prediction = pre.svm_polarity_test(model_dm, dbow_svm_model, dataset='B')


# # https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104
# from pandas import DataFrame, concat
# from doc2vec import LabeledLineSentence
# from sklearn import decomposition
# from measurements import predict
# # import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import multiprocessing
# import doc2vec
# import preprocessing
# import random
# import numpy as np
# import svm
#
#
#
#
# random.seed(1)
# dataset = 'B'
# # test_b = preprocessing.get_data(test, "B")
# # train_b = preprocessing.open_preprocess_file('train', dataset)
# train_b = preprocessing.open_preprocess_file('train', dataset)
#
# # --- get the lables, tweets, and polarities
# topic_lables = train_b['TOPIC']
# text = train_b['CLEANED']
# polarity = train_b['POLARITY']
# # print text
#
# # --- get total of training instances and topics
# num_train = len(topic_lables)
# num_topics = len(topic_lables.value_counts())
# print "total data"
# print num_train
# print "total polarity"
# print polarity.value_counts()
#
# # --- preprocess
# tokens_arr, sents_arr = preprocessing.preprocess(text)
# # print type(sents_arr)
# # sent_topic = topic_lables.to_frame().reset_index(drop=True).join(DataFrame({'TEXT': sents_arr}))
# sent_topic = preprocessing.join_tsp(topic_lables, sents_arr, polarity)
# sentences = list(LabeledLineSentence(sent_topic['TEXT'], sent_topic['TOPIC']))
#
# model_DM = doc2vec.build_doc2vec_model_dm()
# model_DBOW = doc2vec.build_doc2vec_model_dbow()
#
# model_DM.build_vocab(sentences)
# model_DBOW.build_vocab(sentences)
#
#
# # epoch_loop = 200
# epoch_loop = 1
# print ">> -----------------------------"
# print "Doc2Vec training"
# print ">> epoch: " + str(epoch_loop)
# print ">> -----------------------------"
#
# # start training
# for epoch in range(epoch_loop): #200
#     if epoch % 20 == 0:
#         print ('Now training epoch %s' % epoch)
#     # random.shuffle(sentences)
#     model_DM.train(sentences, total_examples=num_train, epochs=3)
#     model_DM.alpha -= 0.002  # decrease the learning rate
#     model_DM.min_alpha = model_DM.alpha  # fix the learning rate, no decay
#
#     model_DBOW.train(sentences, total_examples=num_train, epochs=3)
#     model_DBOW.alpha -= 0.002  # decrease the learning rate
#     model_DBOW.min_alpha = model_DM.alpha  # fix the learning rate, no decay
#
# train_data = concat([sent_topic[sent_topic.POLARITY=='positive'], sent_topic[sent_topic.POLARITY=='negative']]).reset_index(drop=True)
# svm_train_data_dm = doc2vec.build_matrix_csr(model=model_DM, sentences=train_data['TEXT'], topics=train_data['TOPIC'])
# svm_train_data_dbow = doc2vec.build_matrix_csr(model=model_DBOW, sentences=train_data['TEXT'], topics=train_data['TOPIC'])
#
# train_model_dm = svm.split_and_train(svm_train_data_dm, train_data['POLARITY'])
# train_model_dbow = svm.split_and_train(svm_train_data_dbow, train_data['POLARITY'])
#
# # --- get training data
# test_set = preprocessing.open_preprocess_file('test', dataset)
#
# # --- get the lables, tweets, and polarities
# test_polarity = test_set['POLARITY']
# print "total test polarity"
# print test_polarity.value_counts()
#
# test_tokens, test_sents = preprocessing.preprocess(test_set['CLEANED'])
# test_data = preprocessing.join_tsp(test_set['TOPIC'], test_sents, test_set['POLARITY'])
# # test_data = test_data[test_data.POLARITY=='positive'] + test_data[test_data.POLARITY=='negative']
# test_data = concat([test_data[test_data.POLARITY=='positive'], test_data[test_data.POLARITY=='negative']]).reset_index(drop=True)
#
# svm_test_data_dm = doc2vec.build_matrix_csr(model=model_DM, sentences=test_data['TEXT'], topics=test_data['TOPIC'])
# svm_test_data_dbow = doc2vec.build_matrix_csr(model=model_DBOW, sentences=test_data['TEXT'], topics=test_data['TOPIC'])
# # print csr_matrix_test
# # print csr_matrix_test.shape
#
# # --- build svm model >> for polarity
# prediction_res_lda_dm = predict(svm_test_data_dm, test_data['POLARITY'], train_model_dm)
# prediction_res_lda_dbow = predict(svm_test_data_dbow, test_data['POLARITY'], train_model_dbow)
#
#
# def plot_words(w2v):
#     words_np = []
#     # a list of labels (words)
#     words_label = []
#     for word in w2v.vocab.keys():
#         words_np.append(w2v[word])
#         words_label.append(word)
#     print('Added %s words. Shape %s' % (len(words_np), np.shape(words_np)))
#
#     pca = decomposition.PCA(n_components=2)
#     pca.fit(words_np)
#     reduced = pca.transform(words_np)
#
#     # plt.plot(pca.explained_variance_ratio_)
#     for index, vec in enumerate(reduced):
#         # print ('%s %s'%(words_label[index],vec))
#         if index < 100:
#             x, y = vec[0], vec[1]
#             plt.scatter(x, y)
#             plt.annotate(words_label[index], xy=(x, y))
#     plt.show()