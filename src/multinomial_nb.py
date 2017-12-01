from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


def build_multinomial_nb_model(train_vec, train_label, alpha=0.1):
    model = MultinomialNB(alpha=alpha)
    model.fit(train_vec, train_label)
    return model


def split_data(text, label, test_size=0.2, random_state=8):
    text_train, text_test, label_train, label_test = train_test_split(
        text, label, test_size=test_size, random_state=random_state
    )

    return text_train, text_test, label_train, label_test


def split_and_train(matrix, polarity):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # --- build svm model >> for polarity
    svm_model = build_multinomial_nb_model(text_train, pol_train)
    predict(text_test, pol_test, svm_model)
    return svm_model

def predict(text_test, label_test, model):
    prediction = model.predict(text_test)
    print ">> model score: "
    print model.score(text_test, label_test)
    print ">> model report: "
    pol_pre = label_test.to_frame().reset_index(drop=True).join(DataFrame({'PREDICTION': prediction}))
    print pol_pre
    print classification_report(label_test, prediction)
    return pol_pre