from src import features
from src import preprocessing
from src import measurements
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from src.preprocessing import split_data
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from src import text_to_vector
from src import lda
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


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
                    ('selector', ItemSelector(key='text_stat')),
                    ('stats', TextStats()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ])),

            ],

            # weight components in FeatureUnion
            transformer_weights={
                'lda': 0.8,

                'text_stat': 1.0,
            },
        )),
        ('clf',classifier)])
    model=pipeline.fit(train,label)
    return model

#accepts matrix [[sents,topics]]
def split_and_train(matrix,polarity,classifier):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity)
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

        df=x
        text=df['TEXT']
        topics=df['TOPIC']
        num_topics = len(topics.value_counts())
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
        # feature_names = text_to_vector.get_feature_names(vectorizer)

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


        topic_ids = lda.assign_topic_to_ldatopic(vectorizer, lda_model, df)
        topn = dict_len
        topic_words_dist = lda.get_words_topic(lda_model,
                                               topic_ids,
                                               topn
                                               )
        self.model=lda_model
        self.vectorizer=vectorizer
        self.topic_words_dist=topic_words_dist
        self.topic_ids = topic_ids
        self.train_data = df
        return self
    #accepts dictionary from lda param values extractor {text:text,num_topics:num_topics,topic_labels:topic_labels}
    def transform(self,df):
        lda_model=self.model
        # --- get words distribution in for every topic
        vectorizer=self.vectorizer
        topic_words_dist=self.topic_words_dist

        topic_ids = self.topic_ids
        test_set = df

        # print test_data
        test_tokens, test_sents = preprocessing.preprocess(test_set['TEXT'])


        csr_matrix_train = lda.build_matrix_csr(vectorizer=vectorizer,
                                                lda_model=lda_model,
                                                topic_words_dist=topic_words_dist,
                                                map_topic_id=topic_ids,
                                                dataset=test_set
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
    #accept array[[text,topic]]
    def fit(self, x, y=None):
        return self

    def transform(self, matrix):
        # --- get the lables, tweets, and polarities

        topic_lables = []
        text = []
        #polarity=[]
        for r in matrix:
            topic_lables.append(r[1])
            text.append(r[0])
            #polarity.append(r[2])
        # --- get total of training instances and topics
        topic_lables=pd.Series(topic_lables, name="TOPIC")
        text=pd.Series(text,name= "TEXT")
        #polarity=pd.Series(polarity, name ='POLARITY')
        df=pd.concat([topic_lables,text],axis=1)

        #create a dictionary to pass to lda

        fs={}
        fs['lda']=df
        fs['text_stat']=text
        return fs
