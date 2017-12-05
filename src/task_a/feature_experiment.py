from src import preprocessing
from src import pre_task_bc_lda as pre
from src import features
from src import measurements

train_a = preprocessing.open_preprocess_file('train', 'A')
test_a=preprocessing.open_preprocess_file('test', 'A')
tok_list=preprocessing.get_token_for_each_tweet(train_a['CLEANED'])
tok_list_test=preprocessing.get_token_for_each_tweet(test_a['CLEANED'])
#training
X=[]
Y=train_a['POLARITY']
for i in tok_list:
    tokens=i[0]
    word_count=i[1]
    vec= features.build_feature_vector(tokens, word_count)
    X.append(vec)
train_model =pre.classify('logres',X, Y,multi=True, tuning=False)

X_test=[]
Y_test=test_a['POLARITY']
for i in tok_list_test:
    tokens=i[0]
    word_count=i[1]
    vec= features.build_feature_vector(tokens, word_count)
    X_test.append(vec)

prediction = measurements.predict(X_test, Y_test, train_model)
measurements.get_accuracy(prediction)
measurements.avg_recall(prediction)
