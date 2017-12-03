# https://link.springer.com/chapter/10.1007%2F978-3-642-13657-3_43
import text_to_vector
from measurements import predict
import numpy
import lda
import preprocessing
import logistic_regression_multi as logres
import lda_pipeline

dataset = 'B'
train_b = preprocessing.open_preprocess_file('train', dataset)

# --- get the lables, tweets, and polarities
#topic_lables = train_b['TOPIC']
#text = train_b['CLEANED']
#polarity = train_b['POLARITY']

# --- get total of training instances and topics
#num_train = len(train_b)
#num_topics = len(topic_lables.value_counts())
#print "total data"
#print num_train
#print "total polarity"
#print polarity.value_counts()

# --- lda configurations
passes = 20
alpha = 'auto'  # or float number

# --- directory for model and dictionary
# dir_model = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/model/lda_", str(num_train), str(num_topics), str(passes),
#                                       str(alpha))
# dir_model = os.path.abspath(dir_model)
# dir_dict = '{}{}_{}_{}_{}_{}'.format(os.getcwd(), "/dictionary/lda_", str(num_train), str(num_topics), str(passes),
#                                      str(alpha))
# dir_dict = os.path.abspath(dir_dict)

# --- state random
numpy.random.random(1)
lda_pipeline.split_and_train(train_b,logres.initClassifier())
# --- preprocess

