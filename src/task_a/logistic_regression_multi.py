from sklearn.linear_model import LogisticRegression
from measurements import predict
from preprocessing import split_data


def build_log_res_mode(train, label, C=100,multi_class='multinomial',solver='sag'):
    model = LogisticRegression(C=C, multi_class=multi_class,solver=solver).fit(train, label)
    return model


def split_and_train(matrix, polarity,multi_class=False):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # --- build svm model >> for polarity

    model = build_log_res_mode(text_train, pol_train)

    predict(text_test, pol_test, model)
    return model