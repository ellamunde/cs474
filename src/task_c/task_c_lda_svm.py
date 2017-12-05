import src.pre_task_bc_lda as pre

dataset = 'C'
input_from_file = pre.get_data('train', dataset)[:100]
test_set = pre.get_data('test', dataset)[:100]

lda_model, vectorizer, train_data, all_topics, topic_words_dist, topic_ids = pre.get_model(input_from_file)
svm_polarity_model = pre.polarity_model(lda_model=lda_model, model='svm', vectorizer=vectorizer,
                                        topic_words_dist=topic_words_dist, train_data=train_data,
                                        multi=True, map_topic_id=topic_ids)
prediction = pre.polarity_test(lda_model, svm_polarity_model, vectorizer, topic_words_dist, test_set, topic_ids)

import src.measurements as m
m.get_accuracy(prediction)
m.standard_mae(prediction)
m.macro_average_mae(prediction)