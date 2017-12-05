# https://link.springer.com/chapter/10.1007%2F978-3-642-13657-3_43
from src import measurements
from src import preprocessing
from src import logistic_regression as logres
import lda_pipeline
from pandas import concat
dataset = 'B'
train_b = preprocessing.open_preprocess_file('train', dataset)
test_b = preprocessing.open_preprocess_file('test', dataset)
if isinstance(train_b['POLARITY'][0], basestring):
    train_b = concat(
        [train_b[train_b.POLARITY == 'positive'], train_b[train_b.POLARITY == 'negative']]).reset_index(
        drop=True)
matrix=train_b[['CLEANED','TOPIC']].values
polarity=train_b['POLARITY']

model, lda_pipeline_model=lda_pipeline.split_and_train(matrix,polarity)
#testing
matrix=test_b[['CLEANED','TOPIC']].values
polarity=test_b['POLARITY']
prediction=lda_pipeline.predict(model,lda_pipeline_model,matrix,polarity)

measurements.get_accuracy(prediction)
measurements.avg_recall(prediction)

