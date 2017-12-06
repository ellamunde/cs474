from src import features
from src import preprocessing
from src import measurements
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from src.preprocessing import split_data
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from src import pre_task_bc_lda as pre
from sklearn.feature_extraction.text import CountVectorizer


def extractFeatures(train):
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
        ))])
    model=pipeline.fit(train)
    return model


def split_and_train(matrix, polarity,classifier='logres'):
    pipeline_model = extractFeatures(matrix)
    features = pipeline_model.transform(matrix)
    model = pre.classify(classifier, features, polarity, True, False)
    return model,pipeline_model

def predict(model,pipeline_model,test_set,polarity):
    features=pipeline_model.transform(test_set)
    prediction = measurements.predict(features, polarity, model)
    return prediction


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
