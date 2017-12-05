from scipy.sparse import coo_matrix, dok_matrix, hstack, vstack, csr_matrix
# from sklearn.utils import shuffle
# from tqdm import tqdm
# from gensim import corpora
from text_to_vector import transform_text, get_feature_names, convert_to_sparse_bow
import gensim


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


# def process_to_get_topicno(lda_model, bow_words, get='doc'):
#     # print bow_topic
#     doc_topics, word_topics, phi_values = lda_model.get_document_topics(bow_words, per_word_topics=True)
#     if get == 'phi':
#         # return sorted(doc_topics, key=lambda tup: tup[1], reverse=True)[0][0]
#         return sorted(doc_topics, key=lambda tup: tup[1], reverse=True)
#     elif get == 'word':
#         # return sorted(word_topics, key=lambda tup: tup[1], reverse=True)[0][0]
#         return sorted(word_topics, key=lambda tup: tup[1], reverse=True)
#     else:
#         # return sorted(phi_values, key=lambda tup: tup[1], reverse=True)[0][0]
#         return sorted(phi_values, key=lambda tup: tup[1], reverse=True)
#

# def load_lda_model(directory):
#     if os.path.exists(directory):
#         return gensim.models.LdaModel.load(directory)


def build_lda_model(word_bag, dictionary, num_topics, alpha, passes, random_state=0):
    lda_model = gensim.models.ldamodel.LdaModel(word_bag,
                                                num_topics=num_topics,
                                                id2word=dictionary,
                                                passes=passes,
                                                alpha=alpha,
                                                random_state=random_state
                                                )
    print ">> -----------------------------"
    print "LDA model specification:"
    print ">> num_topics: " + str(num_topics)
    print ">> passes: " + str(passes)
    print ">> alpha: " + str(alpha)
    print ">> -----------------------------"

    return lda_model


# def save_lda_model(model, directory):
#     model.save(directory)


def assign_topic_to_ldatopic(vectorizer, lda_model, data):
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

    return final_topicids


def get_list_topics_related(vectorizer, lda_model, data):
    topic_count = {}
    topic_prob = {}
    topics = data['TOPIC']
    texts = data['TEXT']

    # print vectorizer.vocabulary_
    # print ">> lda topics"
    # print lda_model.num_topics
    # print ">> data topics"
    # print len(set(topics))

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

    return matrix_prob


def build_matrix_csr(vectorizer, lda_model, topic_words_dist, map_topic_id, dataset):
    # print dataset
    topics = dataset['TOPIC']
    texts = dataset['TEXT']
    topic_list = {}

    idx_range = len(vectorizer.vocabulary_)
    # print ">> vocabulary range"
    # print idx_range
    len_text = len(topics)
    csrmatrix = None

    for i in range(len_text):
        if topics[i] in map_topic_id.keys():
            topicno = map_topic_id[topics[i]]
        else:
            bow_topic = process_to_bow(vectorizer, lda_model, topics[i])[0]
            topicno = sorted(bow_topic, key=lambda tup: tup[1], reverse=True)[0][0]
            # print ">> bow topic"
            # print bow_topic
            # map_topic_id[topics[i]] = topicno

        text_nth = [0.0] * idx_range
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

        for idx in range(len(column)):
            name = get_feature_names(vectorizer)[idx]
            value[idx] = topic_words_dist[topicno][name] * value[idx] if name in topic_words_dist[topicno].keys() else 0
            # print len(text_nth), " < ", column[idx]
            if len(text_nth) > column[idx]:
                text_nth[column[idx]] = value[idx]

        temp = csr_matrix(text_nth)
        if csrmatrix is not None:
            csrmatrix = vstack([csrmatrix, temp])
        else:
            csrmatrix = temp
        # break

    return csrmatrix

