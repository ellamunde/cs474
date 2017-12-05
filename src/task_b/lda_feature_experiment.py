# https://link.springer.com/chapter/10.1007%2F978-3-642-13657-3_43
import src.text_to_vector
from src.measurements import predict
import numpy
import src.preprocessing as preprocessing
import src.logistic_regression as logres
import lda_pipeline

dataset = 'B'
train_b = preprocessing.open_preprocess_file('train', dataset)[:100]
test_b = preprocessing.open_preprocess_file('test', dataset)[:100]
matrix=train_b[['CLEANED','TOPIC']].values
polarity=train_b['POLARITY']

model=lda_pipeline.split_and_train(matrix,polarity,logres.default_log_res())

matrix=test_b[['CLEANED','TOPIC']].values
polarity=test_b['POLARITY']
measurements.predict(matrix,polarity,model)
