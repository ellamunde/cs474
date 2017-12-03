from sklearn.naive_bayes import MultinomialNB
from measurements import predict
from preprocessing import split_data


def build_multinomial_nb_model(train_vec, train_label, alpha=0.1):
    model = MultinomialNB(alpha=alpha)
    model.fit(train_vec, train_label)
    print ">> -----------------------------"
    print "Naive Bayes Multinoimial model specification:"
    print ">> alpha: " + str(alpha)
    print ">> -----------------------------"
    return model


def split_and_train(matrix, polarity):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # --- build svm model >> for polarity
    model = build_multinomial_nb_model(text_train, pol_train)
    predict(text_test, pol_test, model)
    return model
