"""
Exp14: voting with C-tuned classifiers weighting accoring to mean CV score
"""

import pandas as pd

from cPickle import load

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier

from glove_transformer import GloveVectorizer

# data

data = pd.read_csv(open('semeval2016-task6-trainingdata-utf-8.txt'), '\t',
                   encoding='utf8', index_col=0)
targets = list(data.Target.unique())

# glove

glove_fnames = ('glove_vecs/glove.42B.300d_semeval2016-task6.pkl',
                'glove_vecs/glove.6B.300d_semeval2016-task6.pkl',
                'glove_vecs/glove.840B.300d_semeval2016-task6.pkl',
                'glove_vecs/glove.twitter.27B.200d_semeval2016-task6.pkl'
                )

glove_ids = [fname.split('/')[-1].split('_')[0] for fname in glove_fnames]

glove_vecs = dict((id, pd.read_pickle(fname))
                  for id, fname in zip(glove_ids, glove_fnames))

# construct classifiers

classifiers = dict(
        char_clf=Pipeline([
            ('vect', CountVectorizer(lowercase=False,
                                     min_df=5,
                                     ngram_range=(3, 3),
                                     analyzer='char')),
            ('clf', LinearSVC(class_weight='balanced'))]),

        word_clf=Pipeline([
            ('vect', CountVectorizer(lowercase=False,
                                     ngram_range=(1, 2))),
            ('clf', LinearSVC(class_weight='balanced'))]),

        bigram_clf=Pipeline([
            ('vect', CountVectorizer(stop_words='english',
                                     lowercase=False)),
            ('clf', LinearSVC(class_weight='balanced'))])
)

for id in glove_ids:
    glove_clf = GloveVectorizer(glove_vecs[id])
    classifiers[id] = Pipeline([('vect', glove_clf),
                                ('clf', LinearSVC(class_weight='balanced'))])

vot_clf = VotingClassifier(
        estimators=classifiers.items(),
        # voting='soft',
        # weights=[1, 1, 1]
)

# tuning

C1 = load(open('word_bigram_char_svc_c_tuning.pkl'))
C2 = load(open('glove_svc_c_tuning.pkl'))

# scoring

macro_f_scorer = make_scorer(fbeta_score,
                             beta=1.0,
                             labels=['AGAINST', 'FAVOR'],
                             average='macro')

# exp

for target in targets:
    print 80 * "="
    print target
    print 80 * "="

    target_data = data[data.Target == target]
    true_stances = target_data.Stance
    weights = []

    # set C param and collect weights
    for name, clf in vot_clf.named_estimators.items():
        if name in glove_ids:
            query = "target == '{}' & glove_id == '{}' ".format(target, name)
            row = C2.query(query)
        else:
            query = "target == '{}' & clf == '{}' ".format(target, name)
            row = C1.query(query)

        C = float(row['select_C'])
        clf.set_params(clf__C=C)
        w = float(row['select_mean'])
        weights.append(w)

    # set weight to mean of CV scores for selected C
    vot_clf.set_params(weights=weights)

    # different random state than for tuning C
    cv = StratifiedKFold(true_stances, n_folds=5, shuffle=True, random_state=13)

    scores = cross_val_score(vot_clf, target_data.Tweet, true_stances,
                             scoring=macro_f_scorer, cv=cv)
    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.2f}% (+/- {:.2f})\n'.format(
            scores.mean() * 100, scores.std() * 100)
