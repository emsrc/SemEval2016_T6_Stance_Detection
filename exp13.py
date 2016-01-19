"""
Stacking

Unfinished!...
"""


import pandas as pd
import numpy as np

from cPickle import load

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cross_validation import StratifiedKFold, cross_val_predict
from sklearn.metrics import fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


# %matplotlib inline



def get_results(grid_search):
    results = []

    for t in grid_search.grid_scores_:
        results.append((
            t.mean_validation_score,
            t.cv_validation_scores.std(),
            t.parameters['C']))

    results = pd.DataFrame(results, columns=['score_mean', 'score_std', 'C'])
    results.set_index('C', inplace=True)
    return results


def select_C(results, gamble=0.5):
    """
    select C following the modified "one-standard-error' rule
    """
    best_C = results['score_mean'].idxmax()
    best = results.loc[best_C]
    threshold = best.score_mean - gamble * best.score_std
    select = results[results.score_mean > threshold].iloc[0]
    selected_C = select.name
    return best, select, best_C, selected_C


data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t',
                   index_col=0)
targets = list(data.Target.unique()) #+ ['All']

macro_f_scorer = make_scorer(fbeta_score,
                             beta=1.0,
                             labels=['AGAINST', 'FAVOR'],
                             average='macro')

table = pd.DataFrame(np.zeros(len(targets),
                              dtype=[('target', 'S32'),
                                     ('best_mean', 'f'),
                                     ('best_std', 'f'),
                                     ('select_mean', 'f'),
                                     ('select_std', 'f'),
                                     ('mean_diff', 'f'),
                                     ('std_diff', 'f'),
                                     ('best_C', 'f'),
                                     ('select_C', 'f')]))

base_clfs = dict(

        char_clf=Pipeline([
            ('vect', CountVectorizer(decode_error='ignore',
                                     lowercase=False,
                                     min_df=5,
                                     ngram_range=(3, 3),
                                     analyzer='char')),
            ('clf', LinearSVC(class_weight='balanced'))]),

        word_clf=Pipeline([
            ('vect', CountVectorizer(decode_error='ignore',
                                     lowercase=False,
                                     ngram_range=(1, 2))),
            ('clf', LinearSVC(class_weight='balanced'))]),

        bigram_clf=Pipeline([
            ('vect', CountVectorizer(decode_error='ignore',
                                     stop_words='english',
                                     lowercase=False)),
            ('clf', LinearSVC(class_weight='balanced'))])
)

meta_clf = LinearSVC(class_weight='balanced')

params = dict(
        C=np.logspace(-6, 2, 33)
        # SVC__C=np.hstack([
        #        np.logspace(-6,-2,25),
        #        np.logspace(-1,2,4)])
        # SVC__C=np.logspace(-5,2,15)
        # SVC__C=[0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
)

C1 = load(open('word_bigram_char_svc_c_tuning.pkl'))

i = 0

for target in targets:
    print 80 * "="
    print target
    print 80 * "="

    target_data = data[data.Target == target] #if target != 'All' else data
    # clumsy way of mapping string labels to ints
    true_stances = target_data.Stance  # .str.replace('NONE', '0').replace('FAVOR', '1').replace('AGAINST', '-1')#.astype('int64').as_matrix()

    # different random state than for tuning C
    cv = StratifiedKFold(true_stances, n_folds=5, shuffle=True, random_state=13)

    base_preds = []
    to_float = {'AGAINST': -1.0, 'FAVOR': 1.0, 'NONE': 0}

    for clf_name, clf in base_clfs.items():
        query = "target == '{}' & clf == '{}' ".format(target, clf_name)
        C = C1.query(query)['select_C']
        clf.set_params(clf__C=float(C))
        # FIX: clumsy conversion of labels to floats
        preds = cross_val_predict(clf, target_data.Tweet, true_stances)
        preds = [to_float[l] for l in preds]
        base_preds.append(preds)

    base_preds = np.vstack(base_preds).astype('float64').T

    grid_search = GridSearchCV(meta_clf, params, scoring=macro_f_scorer, cv=cv)
    grid_search.fit(base_preds, true_stances)

    results = get_results(grid_search)
    print results

    # fig, ax = plt.subplots()
    # fig.set_size_inches((15, 10))
    # xlim = ((10e-6, 10e3))
    # results['score_mean'].plot(ax=ax, yerr=results['score_std'], logx=True,
    #                            ylim=(0.2, 0.8), xlim=xlim,
    #                            title=target + ' - ' + vot_clf)

    best, select, best_C, selected_C = select_C(results)
    table.iloc[i] = (target,
                     best.score_mean, best.score_std,
                     select.score_mean, select.score_std,
                     best.score_mean - select.score_mean,
                     best.score_std - select.score_std,
                     best_C, selected_C)
    i += 1


print table

