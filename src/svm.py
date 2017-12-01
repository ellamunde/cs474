from sklearn.svm import SVC
from measurements import predict
from preprocessing import split_data


def train_svm(train, label, C=1000000.0, gamma='auto', kernel='rbf'):
    """
    Create and train the Support Vector Machine.
    """
    # with radial kernel
    svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
    svm_model.fit(train, label)

    return svm_model


def split_and_train(matrix, polarity):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # --- build svm model >> for polarity
    svm_model = train_svm(text_train, pol_train)
    predict(text_test, pol_test, svm_model)
    return svm_model
