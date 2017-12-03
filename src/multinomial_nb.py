from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from measurements import predict
from preprocessing import split_data


def build_nb_model(train_vec, train_label, alpha=0.1, fit_prior=None, multi=True):
    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    model.fit(train_vec, train_label)
    print ">> -----------------------------"
    print "Naive Bayes Multinomial model specification:"
    print ">> alpha: " + str(alpha)
    if fit_prior is not None:
        print ">> alpha: " + str(alpha)
    print ">> -----------------------------"
    print model.get_params(deep=True)
    return model


def tuning_parameters(matrix, polarity, multi=True):
    # Split the dataset in two equal parts
    # xx_train, xx_dev, yy_train, yy_dev = split_data(matrix, polarity, test_size=0.5)

    # Set the parameters by cross-validation
    print "# Tuning hyper-parameters"
    print

    tuned_parameters = [{'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                         'fit_prior': [True, False],
                         }]
    if not multi:
        clf = GridSearchCV((BernoulliNB()), tuned_parameters, cv=5, scoring="precision_macro")
    else:
        clf = GridSearchCV((MultinomialNB()), tuned_parameters, cv=5, scoring="precision_macro")

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
    # return clf.best_estimator_.alpha, clf.best_estimator_.fit_prior
    return clf


def split_and_train(matrix, polarity, multi=True):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity, test_size=0.5)
    print "total polarity split train"
    print pol_train.value_counts()
    print "total polarity split test"
    print pol_test.value_counts()

    # par_alpha, par_fit_prior = tuning_parameters(matrix, polarity)

    # --- build svm model >> for polarity
    # if isinstance(polarity[0], basestring):
    #     model = build_nb_model(text_train, pol_train, alpha=par_alpha, fit_prior=par_fit_prior)
    # else:
    #     model = build_nb_model(text_train, pol_train, alpha=par_alpha, fit_prior=par_fit_prior, multi=False)
    model = tuning_parameters(text_train, pol_train, multi)
    print model.get_params(deep=True)
    predict(text_test, pol_test, model)
    return model
