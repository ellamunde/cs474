from scipy.sparse import coo_matrix, dok_matrix, hstack, vstack, csr_matrix
from sklearn.utils import shuffle
from tqdm import tqdm
from gensim import corpora
from text_to_vector import transform_text, get_feature_names, convert_to_sparse_bow

import gensim
import os


def process_to_bow(vectorizer, lda_model, text):
    # print text
    # print type(text)
    corpus = convert_to_sparse_bow(transform_text(vectorizer, [text]))
    bow_list = [tup for tup in list(lda_model[corpus])]
    return bow_list


# def get_words_topic(lda_model, total_topic, dict_len):
#     topic_words = []
#     for i in tqdm(range(total_topic)):
#         topic_words.append(dict(lda_model.show_topic(topicid=i, topn=dict_len)))
#
#     return topic_words


def get_words_topic(lda_model, topics, topn):
    topic_words = []
    # print topics
    sorted_topics = sorted(topics.items(), key=lambda x:x[1])
    # print topics
    # print sorted_topics

    for k, v in sorted_topics:
        topic_words.append(dict(lda_model.show_topic(topicid=v, topn=topn)))

    return topic_words


def process_to_get_topicno(lda_model, bow_words, get='doc'):
    # print bow_topic
    doc_topics, word_topics, phi_values = lda_model.get_document_topics(bow_words, per_word_topics=True)
    if get == 'phi':
        # return sorted(doc_topics, key=lambda tup: tup[1], reverse=True)[0][0]
        return sorted(doc_topics, key=lambda tup: tup[1], reverse=True)
    elif get == 'word':
        # return sorted(word_topics, key=lambda tup: tup[1], reverse=True)[0][0]
        return sorted(word_topics, key=lambda tup: tup[1], reverse=True)
    else:
        # return sorted(phi_values, key=lambda tup: tup[1], reverse=True)[0][0]
        return sorted(phi_values, key=lambda tup: tup[1], reverse=True)


# def load_lda_model(directory):
#     if os.path.exists(directory):
#         return gensim.models.LdaModel.load(directory)


def build_lda_model(word_bag, dictionary, num_topics, alpha, passes):
    lda_model = gensim.models.ldamodel.LdaModel(word_bag,
                                                num_topics=num_topics,
                                                id2word=dictionary,
                                                passes=passes,
                                                alpha=alpha
                                                )
    print ">> -----------------------------"
    print "LDA model specification:"
    print ">> num_topics: " + str(num_topics)
    print ">> passes: " + str(passes)
    print ">> alpha: " + str(alpha)
    print ">> -----------------------------"

    return lda_model


def save_lda_model(model, directory):
    model.save(directory)


def assign_topic_to_ldatopic(vectorizer, lda_model, data, by_count=True):
    # def set_topic_ldaid(t_value, final_topicids, c_values, topic_ids_count, topic_ids_prob):
    #     for val in c_values:
    #         print val
    #         c_filter = [k[0] for k in topic_ids_count[t_value] if k[1] == val]
    #         print ">> filter by value"
    #         print c_filter
    #
    #         if len(c_filter) > 2:
    #             p_prob = sorted([k[1] for k in topic_ids_prob[t_value] if k[0] in c_filter], reverse=True)
    #             print ">> p_value"
    #             print p_prob
    #
    #             for prob in p_prob:
    #                 print ">> prob"
    #                 print prob
    #                 prob_data = [k for k in topic_ids_prob[t_value] if k[1] == prob and k[0] in c_filter]
    #                 print ">> prob_data"
    #                 print prob_data
    #
    #                 # sort_prob = sorted(c_filter, key=prob_data.__getitem__)
    #                 sort_prob = sorted(prob_data, key=lambda tup: tup[1])
    #                 print ">> sorted prob_data"
    #                 print sort_prob
    #                 for sort in sort_prob:
    #                     print ">> sort in sort"
    #                     print sort
    #                     if sort[0] not in final_topicids.keys():
    #                         print ">> chosen topic"
    #                         print sort
    #                         return sort[0]
    #         elif len(c_filter) == 1:
    #             if c_filter[0] not in final_topicids.keys():
    #                 print ">> chosen topic"
    #                 print c_filter
    #                 return c_filter[0]
    #         else:
    #             p_prob = [k for k in topic_ids_prob[t_value] if k[0] not in final_topicids]
    #             p_prob = sorted(p_prob, key=lambda tup: tup[1], reverse=True)
    #             print ">> else prob"
    #             print p_prob
    #             if len(p_prob) > 0:
    #                 print ">> chosen topic"
    #                 print p_prob
    #                 return p_prob[0][0]


    # shuffle(data)
    total_prob = get_list_topics_related(vectorizer, lda_model, data)
    final_topicids = {}

    # print total_prob
    for t in total_prob.keys():
        for x in range(len(total_prob[t])):
            total_prob[t][x] = total_prob[t][x] / (len(data) * 1.0)
    # print total_prob

    for i in range(lda_model.num_topics):
        # print "topic: " + str(i)
        rank = list()
        for j in total_prob.keys():
            rank.append((j, total_prob[j][i]))

        # print rank
        rank = sorted(rank, key=lambda tup: tup[1], reverse=True)
        # print rank
        for r in rank:
            # print r
            if r[0] not in final_topicids.keys():
                final_topicids[r[0]] = i
                break

    # print final_topicids
    # for t in topic_ids_prob.keys():
    #     topic_ids_prob[t] = {k: topic_ids_prob[t][k]/(topic_ids_count[t][k] * 1.0) for k in topic_ids_prob[t].keys()}
    #
    # for t in topic_ids_prob.keys():
    #     topic_ids_prob[t] = sorted(topic_ids_prob[t].iteritems(), key=lambda (k, v): (v, k))
    #
    # for t in topic_ids_count.keys():
    #     topic_ids_count[t] = sorted(topic_ids_count[t].iteritems(), key=lambda (k, v): (v, k))
    #
    # for t in range(len(set(data['TOPIC']))):
    #     print ">> topic id: " + str(t)
    #     if t in topic_ids_count.keys():
    #         print topic_ids_count[t]
    #         c_values = sorted(set(int(i[1]) for i in topic_ids_count[t] if i[1] not in final_topicids.keys()),
    #                           reverse=True)
    #         # p_values = sorted(set(int(i[1]) for i in topic_ids_prob[t]), reverse=True)
    #         print c_values
    #
    #         final_topicids[t] = set_topic_ldaid(t_value=t, c_values=c_values,
    #                                             topic_ids_count=topic_ids_count,
    #                                             topic_ids_prob=topic_ids_prob,
    #                                             final_topicids=final_topicids
    #                                             )
    #     elif t in topic_ids_prob.keys():
    #         p_prob = [k for k in topic_ids_prob[t] if k[0] not in final_topicids]
    #         print ">> elif prob"
    #         print p_prob
    #         p_prob = sorted(p_prob, key=lambda tup: tup[1], reverse=True)
    #         print p_prob
    #         if len(p_prob) > 0:
    #             print ">> chosen topic"
    #             print p_prob
    #             return p_prob[0][0]

    # print final_topicids
    return final_topicids


def get_list_topics_related(vectorizer, lda_model, data):
    topic_count = {}
    topic_prob = {}
    topics = data['TOPIC']
    texts = data['TEXT']

    # print vectorizer.vocabulary_
    print ">> lda topics"
    print lda_model.num_topics
    print ">> data topics"
    print len(set(topics))

    matrix_prob = {}

    for idx in range(len(topics)):
        # print ">> data >> " + str(idx)
        # print texts[idx]
        # print topics[idx]
        bow = process_to_bow(vectorizer, lda_model, texts[idx])[0]
        # print bow

        belongs_to = lda_model[bow]
        most_sim = sorted(belongs_to, key=lambda belongs_to: belongs_to[1], reverse=True)
        # print most_sim

        # print topics[idx]
        for x in most_sim:
            # print x[0]
            # print x[1]
            if topics[idx] not in matrix_prob.keys():
                matrix_prob[topics[idx]] = [0.0] * lda_model.num_topics

            matrix_prob[topics[idx]][x[0]] += x[1]
        # print matrix_prob[topics[idx]]

        # most_sim = most_sim[0]
        #
        # if most_sim[0] not in topic_count.keys():
        #     topic_count[most_sim[0]] = {}
        #     topic_prob[most_sim[0]] = {}
        #
        # if topics[idx] not in topic_count[most_sim[0]].keys():
        #     topic_count[most_sim[0]][topics[idx]] = 0
        #     topic_prob[most_sim[0]][topics[idx]] = 0.0
        #
        # topic_count[most_sim[0]][topics[idx]] += 1
        # topic_prob[most_sim[0]][topics[idx]] += most_sim[1]

        # temp = list(topic_count[most_sim[0]][topics[idx]])
        # # -- count
        # temp[0] += 1
        # # -- prob
        # temp[1] += most_sim[1]
        # topic_count[most_sim[0]][topics[idx]] = tuple(temp)
        # print topic_count[topics[idx]][most_sim[0]]

    # print topic_count
    # print topic_prob
    # print matrix_prob
    # return topic_count, topic_prob, matrix_prob
    return matrix_prob


def build_matrix_csr(vectorizer, lda_model, topic_words_dist, topics, texts):
    topic_list = {}

    idx_range = len(vectorizer.vocabulary_)
    # print ">> vocabulary range"
    # print idx_range
    len_text = len(topics)

    csrmatrix = None

    for i in tqdm(range(len_text)):
        text_nth = [0.0] * idx_range

        # print ">> topic"
        # print topics[i]

        if topics[i] not in topic_list.keys():
            bow_topic = process_to_bow(vectorizer, lda_model, topics[i])[0]
            topicno = sorted(bow_topic, key=lambda tup: tup[1], reverse=True)[0][0]
            # print ">> bow topic"
            # print bow_topic
            topic_list[topics[i]] = topicno

        topicno = topic_list[topics[i]]
        # print ">> topic no"
        # print topicno

        txt = transform_text(vectorizer, [texts[i]])
        # print type(txt)
        # print ">> matrix"
        # print txt
        txt = txt.tolil(txt)
        value = txt.data[0]
        column = txt.rows[0]
        # print ">> value"
        # print value
        # print ">> column"
        # print column

        for idx in tqdm(range(len(column))):
            name = get_feature_names(vectorizer)[idx]
            # print ">> name"
            # print column[idx]
            # print value[idx]
            # print name

            value[idx] = topic_words_dist[topicno][name] * value[idx] if name in topic_words_dist[topicno].keys() else 0
            # print len(text_nth), " < ", column[idx]

            if len(text_nth) > column[idx]:
                text_nth[column[idx]] = value[idx]

        temp = csr_matrix(text_nth)
        if csrmatrix is not None:
            csrmatrix = vstack([csrmatrix, temp])
        else:
            csrmatrix = temp

        # x_nth.extend([i] * idx_range)
        # y_nth.extend(k for k in range(idx_range))
        # -- FOR DEVELOPMENT ONLY --
        # break

    # print csrmatrix
    return csrmatrix

