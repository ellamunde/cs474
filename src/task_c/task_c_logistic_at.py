from src.measurements import predict

from src import preprocessing

import ord_logistic_regression as logres

import lda_pipeline

from src import preprocessing


import numpy as np
dataset = 'C'

train_c = preprocessing.open_preprocess_file('train', dataset)[:100]
test_c=preprocessing.open_preprocess_file('test', dataset)[:100]

matrix=train_c[['CLEANED','TOPIC']].values
polarity=train_c['POLARITY']

model=lda_pipeline.split_and_train(matrix,polarity,logres.default_log_res())

matrix=test_c[['CLEANED','TOPIC']].values
polarity=test_c['POLARITY']
predict(matrix,polarity,model)



