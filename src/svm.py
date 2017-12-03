from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from measurements import predict
from preprocessing import split_data


def train_svm(train, label, class_weight, c=1000000.0, gamma='auto', kernel='rbf'):
    """
    Create and train the Support Vector Machine.
    """
    # with radial kernel
    svm_model = OneVsRestClassifier(SVC(C=c,
                                        gamma=gamma,
                                        kernel=kernel,
                                        class_weight=class_weight
                                        ))
    # svm_model = SVC(C=C, gamma=gamma, kernel=kernel, )
    svm_model.fit(train, label)

    print ">> -----------------------------"
    print "SVC model specification:"
    print ">> C: " + str(c)
    print ">> gamma: " + str(gamma)
    print ">> kernel: " + str(kernel)
    print ">> -----------------------------"

    return svm_model


def split_and_train(matrix, polarity):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # --- build svm model >> for polarity
    print pol_train
    print text_train
    class_weight = {c: len(text_train[text_train.POLARITY == c]) for c in list(set(pol_train['POLARITY']))}
    svm_model = train_svm(text_train, pol_train, class_weight)
    predict(text_test, pol_test, svm_model)
    return svm_model
