from src import measurements
from src import preprocessing

import pipeline

dataset = 'A'
train = preprocessing.open_preprocess_file('train', dataset)
test = preprocessing.open_preprocess_file('test', dataset)

matrix=train['CLEANED']
polarity=train['POLARITY']

model,pipeline_model=pipeline.split_and_train(matrix,polarity)
#testing
matrix=test['CLEANED']
polarity=test['POLARITY']
prediction=pipeline.predict(model,pipeline_model,matrix,polarity)

measurements.get_accuracy(prediction)
measurements.avg_recall(prediction)

