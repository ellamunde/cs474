from task_a import features
import preprocessing
import measurements
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from preprocessing import split_data
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer


def build_pipeline_mode(train, label,classifier):
    pipeline = Pipeline([
        # Extract the subject & body

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[('tfidf', CountVectorizer(ngram_range=(1, 2),
                                                        # lowercase=True,
                                                        stop_words='english'
                                                        # vocabulary=[(k+1,v) for k,v in dictionary]
                                                        )),
                              # Pipeline for pulling features from the post's subject lin

                              # Pipeline for pulling ad hoc features from post's body
                              ('text_stats', Pipeline([
                                  ('stats', TextStats()),  # returns a list of dicts
                                  ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                              ])),

                              ],

            # weight components in FeatureUnion
            transformer_weights={
                'tfidf': 1.0,
                'text_stats': 1.0,
            },
        )),

        # Use a SVC classifier on the combined features
        ('clf',classifier )
    ])
    model=pipeline.fit(train,label)
    return model


def split_and_train(matrix, polarity,classifier):
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
