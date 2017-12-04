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


def get_words_topic(lda_model, total_topic, dict_len):
    topic_words = []
    for i in tqdm(range(total_topic)):
        topic_words.append(dict(lda_model.show_topic(topicid=i, topn=dict_len)))

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
    # shuffle(data)
    topic_ids_count, topic_ids_prob = get_list_topics_related(vectorizer, lda_model, data)

    final_topic_ids_count = {}
    final_topic_ids_prob = {}
    final_topicids = {}

    for t in topic_ids_prob.keys():
        # topic_ids_prob[] = sorted(topic_ids_count[t].iteritems(), key=lambda (k, v): (v, k))
        topic_ids_prob[t] = {k: topic_ids_prob[t][k]/topic_ids_count[t][k] for k in topic_ids_prob[t].keys()}

    for t in topic_ids_prob.keys():
        # topic_ids_prob[] = sorted(topic_ids_count[t].iteritems(), key=lambda (k, v): (v, k))
        topic_ids_prob[t] = sorted(topic_ids_prob[t].iteritems(), key=lambda (k, v): (v, k))

    for t in topic_ids_count.keys():
        # topic_ids_prob[] = sorted(topic_ids_count[t].iteritems(), key=lambda (k, v): (v, k))
        topic_ids_count[t] = sorted(topic_ids_count[t].iteritems(), key=lambda (k, v): (v, k))


    for t in topic_ids_count.keys():
        print t
        print topic_ids_count[t]
        # c_sort_topic = sorted(topic_ids_count[t].iteritems(), key=lambda (k,v): (v,k))
        # p_sort_topic = sorted(topic_ids_prob[t].iteritems(), key=lambda (k,v): (v,k))

        # c_values = sorted(set(topic_ids_count[t].values()))
        # p_values = sorted(set(topic_ids_prob[t].values()))
        c_values = sorted(set(int(i[1]) for i in topic_ids_count[t]), reverse=True)
        p_values = sorted(set(int(i[1]) for i in topic_ids_prob[t]), reverse=True)

        print c_values
        for val in c_values:
            print val
            c_filter = [k for k, v in topic_ids_count if v == val]
            print c_filter
            if len(c_filter) > 2:
                for prob in p_values:
                    prob_data = {k: v for k, v in topic_ids_prob if v == prob and k in c_filter}
                    sort_prob = sorted(c_filter, key=prob_data.__getitem__)
                    print sort_prob
                    for idx, value in sort_prob:
                        print idx
                        print value

                        # if idx not in final_topicids.keys():
                            # final_topicids[idx] =
                    # # p_filter = [k for k, v in topic_ids_prob.items() if v == prob and k in c_filter]
                    # if len(p_filter) > 1:
                    #     f


        # idx_count = -1
        # count_val = -1
        #
        # idx_prob = -1
        # prob_val = -1
        # for i in topic_ids_count[t].keys():
        #     if topic_ids_count[t][i][0] > count_val:
        #         idx_count = i
        #         count_val = topic_ids[t][i][0]
        #
        #     if topic_ids_count[t][i][1]/topic_ids[t][i][0] > prob_val:
        #         idx_prob = i
        #         prob_val = topic_ids[t][i][1]/topic_ids[t][i][0]


        final_topic_ids_count[t] = idx_count
        final_topic_ids_prob[t] = idx_prob

    print final_topic_ids_count
    print final_topic_ids_prob

    return final_topic_ids_count if by_count else final_topic_ids_prob
        # print sorted(topic_ids[t], key=lambda x: topic_ids[t][0], reverse=True)


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

    for idx in range(len(topics)):
        # print ">> data >> " + str(idx)
        # print texts[idx]
        # print topics[idx]
        bow = process_to_bow(vectorizer, lda_model, texts[idx])[0]
        # print bow

        belongs_to = lda_model[bow]
        most_sim = sorted(belongs_to, key=lambda belongs_to: belongs_to[1], reverse=True)[0]

        if most_sim[0] not in topic_count.keys():
            topic_count[most_sim[0]] = {}
            topic_prob[most_sim[0]] = {}

        if topics[idx] not in topic_count[most_sim[0]].keys():
            topic_count[most_sim[0]][topics[idx]] = 0
            topic_prob[most_sim[0]][topics[idx]] = 0.0

        topic_count[most_sim[0]][topics[idx]] += 1
        topic_prob[most_sim[0]][topics[idx]] += most_sim[1]

        # temp = list(topic_count[most_sim[0]][topics[idx]])
        # # -- count
        # temp[0] += 1
        # # -- prob
        # temp[1] += most_sim[1]
        # topic_count[most_sim[0]][topics[idx]] = tuple(temp)
        # print topic_count[topics[idx]][most_sim[0]]

    print topic_count
    print topic_prob

    return topic_count, topic_prob


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

