from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from src.measurements import predict
from src.preprocessing import split_data
from sklearn.metrics import accuracy_score
import mord

def default_log_res(type='AT'):
    model = mord.LogisticAT(alpha=1.)
    if type == 'SE':
        model = mord.LogisticAT(alpha=1.)
    elif type == 'IT':
        model = mord.LogisticAT(alpha=1.)
    return model

def build_model(train_vec, train_label, type='AT'):
    model = mord.LogisticAT(alpha=1.)
    if type=='SE':
        model = mord.LogisticAT(alpha=1.)
    elif type=='IT':
        model = mord.LogisticAT(alpha=1.)

    model.fit(train_vec, train_label)
    return model



def split_and_train(matrix, polarity, ordlog_type='AT'):
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
    model = build_model(text_train, pol_train,ordlog_type)
    #print model.get_params(deep=True)
    predict(text_test, pol_test, model)
    return model
