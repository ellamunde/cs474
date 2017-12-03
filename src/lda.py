from scipy.sparse import coo_matrix, dok_matrix, hstack, vstack, csr_matrix
from tqdm import tqdm
from gensim import corpora
from text_to_vector import transform_text, get_feature_names, convert_to_sparse_bow

import gensim
import os


def process_to_bow(vectorizer, lda_model, text):
    # print text
    # print type(text)
    corpus = convert_to_sparse_bow(transform_text(vectorizer, [text]))
    bow_list = [tup for tup in tqdm(list(lda_model[corpus]))]

    return bow_list


def get_words_topic(lda_model, total_topic, dict_len):
    topic_words = []
    for i in tqdm(range(total_topic)):
        topic_words.append(dict(lda_model.show_topic(topicid=i, topn=dict_len)))

    return topic_words


def process_to_get_topicno(lda_model, bow_topic, get='doc'):
    # print bow_topic
    doc_topics, word_topics, phi_values = lda_model.get_document_topics(bow_topic, per_word_topics=True)
    if get == 'phi':
        return sorted(doc_topics, key=lambda tup: tup[1], reverse=True)[0][0]
    elif get == 'word':
        return sorted(word_topics, key=lambda tup: tup[1], reverse=True)[0][0]
    else:
        return sorted(phi_values, key=lambda tup: tup[1], reverse=True)[0][0]


def get_dictionary(text):
    # print text
    dictionary = corpora.Dictionary(text)
    # print(dictionary.token2id)
    return dictionary


def load_dictionary(directory):
    return corpora.Dictionary.load(directory)


def load_lda_model(directory):
    if os.path.exists(directory):
        return gensim.models.LdaModel.load(directory)


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

