import pre_task_bc_lda as pre

dataset = 'B'
input_from_file = pre.get_data('train', dataset)[:100]
test_set = pre.get_data('test', dataset)[:100]

lda_model, vectorizer, train_data, all_topics, topic_words_dist = pre.get_model(input_from_file)
logres_polarity_model = pre.polarity_model(lda_model=lda_model, model='logres', vectorizer=vectorizer, topic_words_dist=topic_words_dist, train_data=train_data, multi=False)
prediction = pre.polarity_test(lda_model, logres_polarity_model, vectorizer, topic_words_dist, test_set)

import measurements as m
m.get_accuracy(prediction)
m.avg_recall(prediction)


# # https://link.springer.com/chapter/10.1007%2F978-3-642-13657-3_43
# import text_to_vector
# from measurements import predict
# import numpy
# import lda
# import preprocessing
# import logistic_regression as logres
#
# from pprint import pprint
#
# # --- get training data
# # train_b = preprocessing.get_data('train', 'B')
#
#
# dataset = 'B'
# train_b = preprocessing.open_preprocess_file('train', dataset)
#
# # --- get the lables, tweets, and polarities
# topic_lables = train_b['TOPIC']
# text = train_b['CLEANED']
# polarity = train_b['POLARITY']
#
# # --- get total of training instances and topics
# num_train = len(train_b)
# num_topics = len(topic_lables.value_counts())
# print "total data"
# print num_train
# print "total polarity"
# print polarity.value_counts()
#
# # --- lda configurations
# passes = 20
# alpha = 'auto'  # or float number
#
# # --- directory for model and dictionary
# # dir_model = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/model/lda_", str(num_train), str(num_topics), str(passes),
# #                                       str(alpha))
# # dir_model = os.path.abspath(dir_model)
# # dir_dict = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/dictionary/lda_", str(num_train), str(num_topics), str(passes),
# #                                      str(alpha))
# # dir_dict = os.path.abspath(dir_dict)
#
# # --- state random
# numpy.random.random(1)
#
# # --- preprocess
# tokens_arr, sents_arr = preprocessing.preprocess(text)
#
# # --- init vectorizer
# vectorizer = text_to_vector.count_vectorizer()
# # vectorizer = lda.tfidf_vectorizer()
# # vectorizer = lda.fit_to_vectorizer(vectorizer, sents_arr)
#
# # --- convert text into vectors using vectorizer
# bow_vectorizer = text_to_vector.fit_to_vectorizer(vectorizer, sents_arr)
# # print ">> bow"
# # print bow_vectorizer
#
# # --- get feature names based on n-grams
# feature_names = text_to_vector.get_feature_names(vectorizer)
#
# # --- convert dictionary to id2word
# idvec2word = text_to_vector.map_idvec2word(vectorizer)
# dict_len = len(idvec2word)
#
# # --- convert bow vectorizer into bow lda
# bow_lda = text_to_vector.convert_to_sparse_bow(bow_vectorizer)
# # print ">> bow lda"
# # print bow_lda
#
# # --- build lda model >> for topic
# lda_model = lda.build_lda_model(word_bag=bow_lda,
#                                 dictionary=idvec2word,
#                                 num_topics=num_topics,
#                                 alpha=alpha,
#                                 passes=passes
#                                 )
#
# # --- get words distribution in for every topic
# topic_words_dist = lda.get_words_topic(lda_model, num_topics, dict_len)
#
# csr_matrix_train = lda.build_matrix_csr(vectorizer=vectorizer,
#                                         lda_model=lda_model,
#                                         topic_words_dist=topic_words_dist,
#                                         topics=topic_lables,
#                                         texts=text
#                                         )
#
# # --- split data for svm training and test
# # text_train, text_test, pol_train, pol_test = svm.split_data(bow_vectorizer, polarity)
# # text_train, text_test, pol_train, pol_test = svm.split_data(csr_matrix_train, polarity)
#
# # --- build svm model >> for polarity
# # svm_model = svm.train_svm(text_train, pol_train)
# # svm.predict(text_test, pol_test, svm_model)
#
# # train_model = logres.split_and_train(bow_vectorizer, polarity)
# train_model_lda = logres.split_and_train(csr_matrix_train, polarity)
#
# ## --- to get all the topics from lda
# # all_topics = lda_model.get_document_topics(bow=bow_lda, per_word_topics=True)
#
# ##################################################################################################################
# ################################################### EXPERIMENT ###################################################
# ##################################################################################################################
#
# # for i in range(len(topic_words_dist)):
# #     print i
# #     print topic_words_dist[i]
#
# # --- get training data
# test_set = preprocessing.open_preprocess_file('test', dataset)
#
# # --- get the lables, tweets, and polarities
# test_topics = test_set['TOPIC']
# test_text = test_set['CLEANED']
# test_polarity = test_set['POLARITY']
# len_test = len(test_text)
# # print ">> topic"
# # print len_test
#
# print "total test polarity"
# print test_polarity.value_counts()
#
# test_tokens, test_sents = preprocessing.preprocess(test_text)
# topic_list = {}
#
# csr_matrix_test = lda.build_matrix_csr(vectorizer=vectorizer,
#                                        lda_model=lda_model,
#                                        topic_words_dist=topic_words_dist,
#                                        topics=test_topics,
#                                        texts=test_text
#                                        )
# # print csr_matrix_test
# # print csr_matrix_test.shape
#
# # --- build svm model >> for polarity
# # prediction_res = predict(csr_matrix_test, test_polarity, train_model)
# prediction_res_lda = predict(csr_matrix_test, test_polarity, train_model_lda)
