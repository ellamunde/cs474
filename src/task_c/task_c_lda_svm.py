import pre_task_bc_lda as pre

input_from_file = pre.get_data('train', 'C')[:100]
lda_model, vectorizer, train_data, all_topics, topic_words_dist = pre.get_model(input_from_file)
svm_polarity_model = pre.svm_polarity_model(lda_model, vectorizer, topic_words_dist, train_data)
prediction = pre.svm_polarity_test(lda_model, svm_polarity_model, vectorizer, topic_words_dist, dataset='C')
