from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from measurements import predict
from preprocessing import split_data


def default_log_res():
    return LogisticRegression(C=100)


def build_log_res_model(train, label, c=100, solver='liblinear', tol=None, multi_class=None,
                        fit_intercept=None, class_weight=None):
    print ">> -----------------------------"
    print "Logistic Regression model specification:"
    print ">> C: " + str(c)
    print ">> solver: " + solver
    if tol is not None:
        print ">> tol: " + str(tol)
    if class_weight is not None:
        print ">> class_weight: " + class_weight
    print ">> fit_intercept: " + str(fit_intercept)
    print ">> -----------------------------"
    if multi_class is None:
        return LogisticRegression(C=c, solver=solver, tol=tol, class_weight=class_weight, fit_intercept=fit_intercept).fit(
        train, label)
    else:
        return LogisticRegression(C=c, solver=solver, tol=tol, class_weight=class_weight, multi_class=multi_class,
                                  fit_intercept=fit_intercept).fit(train, label)


def tuning_parameters(matrix, polarity):
    # Split the dataset in two equal parts
    # xx_train, xx_dev, yy_train, yy_dev = split_data(matrix, polarity, test_size=0.5)

    # Set the parameters by cross-validation
    if isinstance(polarity[0], basestring):
        tuned_parameters = [{'tol': [1e-3, 1e-4], 'solver': ['liblinear'],
                             'C': [1, 10, 100, 1000, 10000, 100000], 'fit_intercept': [True, False],
                             'class_weight': [None, 'balanced']}]
    else:
        tuned_parameters = [{'tol': [1e-3, 1e-4], 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                             'C': [1, 10, 100, 1000, 10000, 100000], 'fit_intercept': [True, False],
                             'class_weight': [None, 'balanced'], 'multi_class': ['multinomial', 'ovr']}]

    print "# Tuning hyper-parameters"
    print
    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5)
    clf.fit(matrix, polarity)
    print "Best parameters set found on development set:"
    print clf.best_estimator_
    print
    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    # predict(xx_dev, yy_dev, clf)
    # return clf.best_estimator_.C, clf.best_estimator_.solver, clf.best_estimator_.tol, clf.best_estimator_.class_weight, \
    #        clf.best_estimator_.multi_class, clf.best_estimator_.fit_intercept
    return clf


def split_and_train(matrix, polarity, tuning=True):
    # print matrix
    # print polarity
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity, test_size=0.5)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # Standarize features
    scaler = StandardScaler(with_mean=False)
    text_train_std = scaler.fit_transform(text_train)
    # text_train_std = text_train

    # if tuning:
    #     par_c, par_solver, par_tol, par_c_weight, par_multi_c, par_intercept = tuning_parameters(matrix, polarity)
    #     # --- build svm model >> for polarity
    #     model = build_log_res_model(text_train, pol_train, par_c, par_solver, par_tol, par_multi_c, par_intercept, par_c_weight)
    # else:
    #     model = default_log_res()
    if tuning:
        model = tuning_parameters(text_train, pol_train)
    else:
        model = default_log_res()
    print model.get_params(deep=True)
    predict(text_test, pol_test, model)
    return model
