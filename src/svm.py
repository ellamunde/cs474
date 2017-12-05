from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC
from measurements import predict
from preprocessing import split_data
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight
import numpy as np


# def train_svm(train, label, class_weight, c=1000000.0, gamma='auto', kernel='rbf'):
#     """
#     Create and train the Support Vector Machine.
#     """
#     # with radial kernel
#     svm_model = OneVsRestClassifier(SVC(C=c,
#                                         gamma=gamma,
#                                         kernel=kernel,
#                                         class_weight=class_weight
#                                         ))
#     # svm_model = SVC(C=C, gamma=gamma, kernel=kernel, )
#     svm_model.fit(train, label)
#
#     print ">> -----------------------------"
#     print "SVC model specification:"
#     print ">> C: " + str(c)
#     print ">> gamma: " + str(gamma)
#     print ">> kernel: " + str(kernel)
#     print ">> -----------------------------"
#
#     print svm_model.get_params(deep=True)
#
#     return svm_model


def tuning_parameter(matrix, polarity, multi=True):
    # Split the dataset in two equal parts
    # xx_train, xx_dev, yy_train, yy_dev = split_data(matrix, polarity, test_size=0.5)

    # Set the parameters by cross-validation
    scoring = {#'auc': 'roc_auc',
               'accuracy': make_scorer(accuracy_score),
               # 'neg_mean_squared_error': 'neg_mean_squared_error'
               'precision': 'precision',
               'precision_macro': 'precision_macro'
               }
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity, test_size=0.2)

    print "# Tuning hyper-parameters"
    print
    if not multi:
        tuned_parameters = [{'kernel': ['rbf', 'linear'],
                             'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000, 10000, 100000],
                             'class_weight': [None, 'balanced']
                             }]

        for a_class in set(polarity):
            # y_this_class = (polarity == a_class)
            y_this_class = (pol_train == a_class)
            model_to_tune = GridSearchCV(SVC(random_state=0), tuned_parameters, cv=5,
                                         scoring=scoring, refit='precision_macro')
            # model_tuned = GridSearchCV(model_to_tune, param_grid=params, scoring='f1', n_jobs=2)
            # model_to_tune.fit(matrix, y_this_class)
            model_to_tune.fit(text_train, y_this_class)

            for i in model_to_tune.best_params_.keys():
                if i not in tuned_parameters[0].keys():
                    tuned_parameters[0][i] = []
                elif i in tuned_parameters[0][i]:
                    continue

                tuned_parameters[0][i].append(model_to_tune.best_params_[i])

        # clf = GridSearchCV((SVC()), tuned_parameters, cv=5, scoring='precision')
        clf = GridSearchCV((SVC(random_state=0)), tuned_parameters, cv=5)
        # if len(set(polarity)) > 2:
        #     clf = GridSearchCV((SVC(random_state=0)), tuned_parameters, cv=5)
        # else:
        #     clf = GridSearchCV((SVC(random_state=0)), tuned_parameters, cv=5, scoring='precision')
    else:
        tuned_parameters = [{'estimator__kernel': ['rbf', 'linear'],
                             'estimator__gamma': [1e-3, 1e-4],
                             'estimator__C': [1, 10, 100, 1000, 10000, 100000],
                             'estimator__class_weight': [None, 'balanced']
                             }]

        # for a_class in set(polarity):
        #     # y_this_class = (polarity == a_class)
        #     y_this_class = (pol_train == a_class)
        #     model_to_tune = GridSearchCV(OneVsRestClassifier(SVC(random_state=0)),
        #                                  tuned_parameters, cv=5,
        #                                  scoring=scoring, refit='precision_macro')
        #     # model_tuned = GridSearchCV(model_to_tune, param_grid=params, scoring='f1', n_jobs=2)
        #     # model_to_tune.fit(matrix, y_this_class)
        #     model_to_tune.fit(text_train, y_this_class)
        #     # print model_to_tune.best_params_
        #
        #     for i in model_to_tune.best_params_.keys():
        #         if i not in tuned_parameters[0].keys():
        #             tuned_parameters[0][i] = []
        #         elif i in tuned_parameters[0][i]:
        #             continue
        #
        #         tuned_parameters[0][i].append(model_to_tune.best_params_[i])

        clf = GridSearchCV(OneVsRestClassifier(SVC(random_state=0)), tuned_parameters, cv=5)
        # if len(set(polarity)) > 2:
        #     clf = GridSearchCV(OneVsRestClassifier(SVC(random_state=0)), tuned_parameters, cv=5)
        # else:
        #     clf = GridSearchCV(OneVsRestClassifier(SVC(random_state=0)), tuned_parameters, cv=5, scoring=scoring, refit='precision')
    # print chosen_par
    # clf.fit(matrix, polarity)
    clf.fit(text_train, pol_train)

    print "Best parameters set found on development set:"
    print clf.best_estimator_
    print
    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    predict(text_test, pol_test, clf)
    # if isinstance(polarity[0], basestring):
    #     return clf.best_estimator_.C, clf.best_estimator_.kernel, clf.best_estimator_.gamma, clf.best_estimator_.class_weight
    # return clf.best_estimator_.estimator__C, clf.best_estimator_.estimator__kernel, clf.best_estimator_.estimator__gamma, clf.best_estimator_.estimator__class_weight
    # print clf.best_estimator_
    # print clf.best_params_
    # return clf.best_params_['estimator__C'], clf.best_params_['estimator__kernel'], \
    #     clf.best_params_['estimator__gamma'], clf.best_params_['estimator__class_weight']
    return clf

    # scores = ['precision', 'recall']
    # for score in scores:
    #     print "# Tuning hyper-parameters for %s" % score
    #     print
    #
    #     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
    #                        scoring='%s_macro' % score)
    #
    #     clf.fit(X_train, y_train)
    #     print "Best parameters set found on development set:"
    #     print clf.best_estimator_
    #     print
    #     print "Grid scores on development set:"
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    #     print
    #     print "Detailed classification report:"
    #     print
    #     print "The model is trained on the full development set."
    #     print "The scores are computed on the full evaluation set."
    #     print
    #
    #     # y_true, y_pred = y_test, clf.predict(X_test)
    #     predict(X_test, y_test, clf)


def split_and_train(matrix, polarity, multi=True):
    text_train, text_test, pol_train, pol_test = split_data(matrix, polarity, test_size=0.5)

    print "total polarity split train"
    pol_train_pol = pol_train.value_counts()
    print pol_train_pol
    print "total polarity split test"
    print pol_test.value_counts()
    total = len(pol_train) * 1.0

    # --- build svm model >> for polarity
    # y_class = mlb.fit_transform([[k] for k in pol_train_pol.index])
    # class_weight = {k: pol_train_pol[k]/total for k in pol_train_pol.index}
    # print "class weight"
    # print type(class_weight)
    # print class_weight
    # c_weight = compute_class_weight(class_weight, np.unique(pol_train_pol.index), pol_train_pol.index)
    # print c_weight
    # svm_model = train_svm(text_train, pol_train, class_weight)

    # @optunity.cross_validated(x=matrix, y=polarity, num_folds=10, num_iter=2)
    # def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    #     model = SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    #     decision_values = model.decision_function(x_test)
    #     return optunity.metrics.roc_auc(y_test, decision_values)
    #
    # # perform tuning
    # hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])
    # train model on the full training set with tuned hyperparameters
    # optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(data, labels)
    # svm_model = train_svm(text_train, pol_train, 'balanced', c=10 ** hps['logC'], gamma=10 ** hps['logGamma'])

    # Standarize features
    scaler = StandardScaler(with_mean=False)
    # text_train_std = scaler.fit_transform(text_train)
    text_train_std = text_train


    # par_c, par_kernel, par_gamma, par_c_weight = tuning_parameter(text_train_std, pol_train)
    # par_c, par_kernel, par_gamma, par_c_weight = tuning_parameter(matrix, polarity)
    # svm_model = train_svm(text_train_std,
    #                       pol_train,
    #                       class_weight=par_c_weight,
    #                       c=par_c,
    #                       kernel=par_kernel,
    #                       gamma=par_gamma
    #                       )
    # svm_model = train_svm(text_train, pol_train, None)
    # print ">> classes"
    # print svm_model.classes_
    # print svm_model.n_classes_

    svm_model = tuning_parameter(text_train_std, pol_train, multi)
    # print svm_model.get_params(deep=True)
    predict(text_test, pol_test, svm_model)
    return svm_model
