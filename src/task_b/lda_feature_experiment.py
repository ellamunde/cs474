# https://link.springer.com/chapter/10.1007%2F978-3-642-13657-3_43
import text_to_vector
from measurements import predict
import numpy
import lda
import preprocessing
import logistic_regression as logres
import lda_pipeline

dataset = 'B'
train_b = preprocessing.open_preprocess_file('train', dataset)

# --- lda configurations
passes = 20
alpha = 'auto'  # or float number

# --- state random
numpy.random.random(1)
lda_pipeline.split_and_train(train_b, logres.default_log_res())
