# https://link.springer.com/chapter/10.1007%2F978-3-642-13657-3_43
import numpy
import lda
import preprocessing
import svm

from pprint import pprint

# --- get training data
# train_b = preprocessing.get_data('train', 'B')
import text_to_vector
from measurements import predict

dataset = 'B'
train_b = preprocessing.open_preprocess_file('train', dataset)

# --- get the lables, tweets, and polarities
topic_lables = train_b['TOPIC']
text = train_b['CLEANED']
polarity = train_b['POLARITY']

# --- get total of training instances and topics
num_train = len(train_b)
num_topics = len(topic_lables.value_counts())
print "total data"
print num_train
print "total polarity"
print polarity.value_counts()

# --- lda configurations
passes = 20
alpha = 'auto'  # or float number

# --- directory for model and dictionary
# dir_model = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/model/lda_", str(num_train), str(num_topics), str(passes),
#                                       str(alpha))
# dir_model = os.path.abspath(dir_model)
# dir_dict = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/dictionary/lda_", str(num_train), str(num_topics), str(passes),
#                                      str(alpha))
# dir_dict = os.path.abspath(dir_dict)

# --- state random
numpy.random.random(1)

# --- preprocess
tokens_arr, sents_arr = preprocessing.preprocess(text)

# --- init vectorizer
vectorizer = text_to_vector.count_vectorizer()
# vectorizer = lda.tfidf_vectorizer()
# vectorizer = lda.fit_to_vectorizer(vectorizer, sents_arr)

# --- convert text into vectors using vectorizer
bow_vectorizer = text_to_vector.fit_to_vectorizer(vectorizer, sents_arr)
# print ">> bow"
# print bow_vectorizer

# --- get feature names based on n-grams
feature_names = text_to_vector.get_feature_names(vectorizer)

# --- convert dictionary to id2word
idvec2word = text_to_vector.map_idvec2word(vectorizer)
dict_len = len(idvec2word)

# --- convert bow vectorizer into bow lda
bow_lda = text_to_vector.convert_to_sparse_bow(bow_vectorizer)
# print ">> bow lda"
# print bow_lda

# --- build lda model >> for topic
lda_model = lda.build_lda_model(word_bag=bow_lda,
                                dictionary=idvec2word,
                                num_topics=num_topics,
                                alpha=alpha,
                                passes=passes
                                )

def get_features():
    """
    1.  number of words
    2.  probability of pos
    3.  probability of neg
    4.  probability of adj
    5.  probability of v
    6.  probability of n
    7.  number of link
    8.  number of mention
    9.  number of emoticon
    10. number of numbers
    11. boolean positive intensifier
    12. boolean negative intensifier
    13. boolean positive negation
    14. boolean negative negation
    15. boolean future tense
    16. boolean has modal
    17. boolean has pronoun
    18. boolean
    :return:
    """
    number_of_words=word_count
    #fractions with respect to number of words in a tweet
    fraction_pos,fraction_neg=get_tokens_fractions(tweet,number_of_words)
    fraction_of_adj,fraction_of_v,fraction_of_n=get_pos_fractions(tweet,number_of_words)
    has_link=len([l for l,tag,pol in tweet if l=='%link'])
    has_mention=len([m for m,tag,pol in tweet if m=='%mention'])
    has_emoticon=len([e for e,tag,pol in tweet if e.startswith('%emoticon')])
    has_number=len([n for n,tag,pol in tweet if n=='%number'])
    #intensifier followed by positives
    intens_followed_by_pos,intens_followed_by_neg,intens_followed_by_neu=check_intens(tweet)
    #negation of positive or negative
    neg_followed_by_pos, neg_followed_by_neg,neg_followed_by_neu = check_negation(tweet)
    future_tense=check_tense(tweet)
    has_modal=check_modal(tweet)
    has_1_person=check_pronoun(tweet)
    contains_pos=containsPos(tweet)
    contains_neg = containsNeg(tweet)
    pos_postion,neg_postion=neg_pos_position(tweet)
    has_excl=len([t for t,tag,pol in tweet if t=="%exclamation"])
    has_stop=len([t for t,tag,pol in tweet if t=="%fullstop"])
    sample=[number_of_words,fraction_pos,fraction_neg,
            fraction_of_adj,fraction_of_n,fraction_of_v,
            has_link,has_mention,has_emoticon,has_number,intens_followed_by_pos,
            intens_followed_by_neg,intens_followed_by_neu,neg_followed_by_pos, neg_followed_by_neg,
            neg_followed_by_neu,future_tense,has_modal,has_1_person,contains_pos,contains_neg,
            pos_postion,neg_postion,has_excl,has_stop]
    return sample