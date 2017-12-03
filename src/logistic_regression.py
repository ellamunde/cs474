from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from measurements import predict
from preprocessing import split_data


def initClassifier():
    return LogisticRegression(C=100)


def build_log_res_model(train, label, C=100):
    print ">> -----------------------------"
    print "Logistic Regression model specification:"
    print ">> C: " + str(C)
    print ">> -----------------------------"
    return LogisticRegression(C=C).fit(train, label)


def tuning_parameters(matrix, polarity):
    # Split the dataset in two equal parts
    xx_train, xx_dev, yy_train, yy_dev = split_data(matrix, polarity, test_size=0.5)

    # Set the parameters by cross-validation
    tuned_parameters = [{'tol': [1e-3, 1e-4], 'solver': ['liblinear'],
                         'C': [1, 10, 100, 1000, 10000, 100000], 'fit_intercept': [True, False],
                         'class_weight': [None, 'balanced'], 'multi_class': ['multinomial']}]

    print "# Tuning hyper-parameters"
    print
    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5)
    clf.fit(xx_train, yy_train)
    print "Best parameters set found on development set:"
    print clf.best_estimator_
    print
    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    result = predict(xx_dev, yy_dev, clf)
    return clf.best_estimator_.C, clf.best_estimator_.kernel, clf.best_estimator_.gamma, clf.best_estimator_.class_weight


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



