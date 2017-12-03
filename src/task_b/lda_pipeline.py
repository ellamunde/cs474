import features
import preprocessing
import measurements
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from preprocessing import split_data
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import text_to_vector
import lda


def build_pipeline_mode(train, label,classifier):

    pipeline = Pipeline([
        # Extract the parameters for lda and feature
        ('ldaextractor', LdaParamValuesExtractor()),

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features from the post's subject line
                ('lda', Pipeline([
                    ('selector', ItemSelector(key='lda')),
                    ('ldavec', LdaVec()),
                ])),

                # Pipeline for pulling ad hoc features from post's body
                ('text_stats', Pipeline([
                    ('selector', ItemSelector(key='text_stats')),
                    ('stats', TextStats()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ])),

            ],

            # weight components in FeatureUnion
            transformer_weights={
                'lda': 0.8,

                'text_stats': 1.0,
            },
        )),

        # Use a SVC classifier on the combined features
        ('clf', classifier ),
    ])

    model=pipeline.fit(train,label)
    return model

#accepts dataframe
def split_and_train(df,classifier):
    polarity=df['POLARITY']
    text_train, text_test, pol_train, pol_test = split_data(df, polarity)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # --- build svm model >> for polarity

    model = build_pipeline_mode(text_train, pol_train,classifier)

    measurements.predict(text_test, pol_test, model)
    return model


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        tok_list = preprocessing.get_token_for_each_tweet(posts)
        X = []

        for i in tok_list:
            tokens = i[0]
            word_count = i[1]


            sample = features.build_feature_vector(tokens, word_count)
            dict = features.convert_to_dict(sample)

            X.append(dict)
        return X

class LdaVec(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self
    #accepts dictionary from lda param values extractor {text:text,num_topics:num_topics,topic_labels:topic_labels}
    def transform(self, lda_dict):
        text=lda_dict['text']
        num_topics=lda_dict['num_topics']
        topic_lables=lda_dict['topic_labels']
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
                                        alpha='auto',
                                        passes=20
                                        )

        # --- get words distribution in for every topic
        topic_words_dist = lda.get_words_topic(lda_model, num_topics, dict_len)

        csr_matrix_train = lda.build_matrix_csr(vectorizer=vectorizer,
                                                lda_model=lda_model,
                                                topic_words_dist=topic_words_dist,
                                                topics=topic_lables,
                                                texts=text
                                                )
        return  csr_matrix_train
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class LdaParamValuesExtractor(BaseEstimator, TransformerMixin):
    """Extract the info necessary for lda & text stats from a dataframe in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `lda` and `text_stat`.
    """
    #accept dataframe
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        # --- get the lables, tweets, and polarities
        topic_lables = df['TOPIC']
        text = df['CLEANED']

        # --- get total of training instances and topics
        num_topics = len(topic_lables.value_counts())
        #create a dictionary to pass to lda
        dic_lda={'text': text, 'topic_labels':topic_lables,'num_topics':num_topics}
        fs={}
        fs['lda']=dic_lda
        fs['text_stat']=text
        return fs
