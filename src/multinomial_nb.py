from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from measurements import predict
from preprocessing import split_data
from sklearn.metrics import accuracy_score


# def build_nb_model(train_vec, train_label, alpha=0.1, fit_prior=None, multi=True):
#     model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
#     model.fit(train_vec, train_label)
#     print ">> -----------------------------"
#     print "Naive Bayes Multinomial model specification:"
#     print ">> alpha: " + str(alpha)
#     if fit_prior is not None:
#         print ">> alpha: " + str(alpha)
#     print ">> -----------------------------"
#     print model.get_params(deep=True)
#     return model


def tuning_parameters(matrix, polarity, multi=True):
    # Split the dataset in two equal parts
    # xx_train, xx_dev, yy_train, yy_dev = split_data(matrix, polarity, test_size=0.5)

    # Set the parameters by cross-validation
    print "# Tuning hyper-parameters"
    print

    scoring = {#'auc': 'roc_auc',
               'accuracy': make_scorer(accuracy_score),
               # 'neg_mean_squared_error': 'neg_mean_squared_error'
               'precision': 'precision',
               'precision_macro': 'precision_macro'
               }

    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity, test_size=0.2)

    tuned_parameters = [{'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                         'fit_prior': [True, False],
                         }]
    if not multi:
        for a_class in set(polarity):
            # y_this_class = (polarity == a_class)
            y_this_class = (pol_train == a_class)
            model_to_tune = GridSearchCV(BernoulliNB(), tuned_parameters, cv=5,
                                         scoring=scoring, refit='precision_macro')
            # model_tuned = GridSearchCV(model_to_tune, param_grid=params, scoring='f1', n_jobs=2)
            model_to_tune.fit(text_train, y_this_class)

            for i in model_to_tune.best_params_.keys():
                if i not in tuned_parameters[0].keys():
                    tuned_parameters[0][i] = []
                elif i in tuned_parameters[0][i]:
                    continue

                tuned_parameters[0][i].append(model_to_tune.best_params_[i])

        clf = GridSearchCV((BernoulliNB()), tuned_parameters, cv=5)
    else:
        for a_class in set(polarity):
            # y_this_class = (polarity == a_class)
            y_this_class = (pol_train == a_class)
            model_to_tune = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5,
                                         scoring=scoring, refit='precision_macro')
            # model_tuned = GridSearchCV(model_to_tune, param_grid=params, scoring='f1', n_jobs=2)
            # model_to_tune.fit(matrix, y_this_class)
            model_to_tune.fit(text_train, y_this_class)

            # for i in model_to_tune.best_params_.keys():
            #     if i not in tuned_parameters[0].keys():
            #         tuned_parameters[0][i] = []
            #     elif i in tuned_parameters[0][i]:
            #         continue
            #
            #     tuned_parameters[0][i].append(model_to_tune.best_params_[i])

        clf = GridSearchCV((MultinomialNB()), tuned_parameters, cv=5)

    # clf.fit(matrix, polarity)
    clf.fit(text_train, pol_train)
    # if not multi:
    #     # clf = GridSearchCV((BernoulliNB()), tuned_parameters, cv=5, scoring=scoring, refit='auc')
    #     clf = GridSearchCV((BernoulliNB()), tuned_parameters, cv=5, scoring='accuracy')
    # else:
    #     clf = GridSearchCV((MultinomialNB()), tuned_parameters, cv=5, scoring='accuracy')

    # if len(set(polarity)) > 2:
    #     for a_class in set(polarity):
    #         y_this_class = (polarity == a_class)
    #         # model_tuned = GridSearchCV(model_to_tune, param_grid=params, scoring='f1', n_jobs=2)
    #         clf.fit(matrix, y_this_class)
    # else:
    #     clf.fit(matrix, polarity)

    print "Best parameters set found on development set:"
    print clf.best_estimator_
    print
    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    predict(text_test, pol_test, clf)
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
