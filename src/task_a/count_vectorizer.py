from src import text_to_vector
from src import measurements
import numpy
from src import preprocessing
from src import pre_task_bc_lda as pre
train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"
train_a = preprocessing.open_preprocess_file(train, "A")
test_a = preprocessing.open_preprocess_file(test, "A")

text = train_a['CLEANED']
polarity = train_a['POLARITY']
test=test_a['CLEANED']
polarity_t=test_a['POLARITY']


# --- preprocess
tokens_arr, sents_arr = preprocessing.preprocess(text)
vectorizer = text_to_vector.count_vectorizer()

bow_vectorizer = text_to_vector.fit_to_vectorizer(vectorizer, sents_arr)

train_model =pre.classify('logres',bow_vectorizer, polarity,multi=True, tuning=False)
tokens_arr, sents_arr = preprocessing.preprocess(test)
bow_vectorizer = text_to_vector.transform_text(vectorizer, sents_arr)
prediction = measurements.predict(bow_vectorizer, polarity_t, train_model)
measurements.get_accuracy(prediction)
measurements.avg_recall(prediction)
