from sklearn.linear_model import LogisticRegression
from measurements import predict
from preprocessing import split_data


def build_log_res_model(train, label, C=100):
    model = LogisticRegression(C=C).fit(train, label)
    return model


def split_and_train(matrix, polarity):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # --- build svm model >> for polarity
    model = build_log_res_model(text_train, pol_train)
    predict(text_test, pol_test, model)
    return model



