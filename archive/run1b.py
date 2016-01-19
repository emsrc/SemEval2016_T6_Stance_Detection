"""
Run1 based on
Exp14b: voting with C-tuned classifiers
"""

import csv
from cPickle import load

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, fbeta_score

from glove_transformer import GloveVectorizer

# data

train_data = pd.read_csv(open('semeval2016-task6-trainingdata-utf-8.txt'), '\t',
                         encoding='utf8',
                         index_col=0)
targets = list(train_data.Target.unique())

test_data = pd.read_csv(open('SemEval2016-Task6-subtaskA-testdata.txt'), '\t',
                        encoding='utf8',
                        index_col=0)

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

for id in glove_ids:
    glove_clf = GloveVectorizer(glove_vecs[id])
    classifiers[id] = Pipeline([('vect', glove_clf),
                                ('clf', LinearSVC(class_weight='balanced'))])

vot_clf = VotingClassifier(
        estimators=classifiers.items(),
        # voting='soft',
        # weights=[1, 1, 1]
)

# C tuning

C1 = load(open('word_bigram_char_svc_c_tuning.pkl'))
C2 = load(open('glove_svc_c_tuning.pkl'))

# run

true_train_stances = train_data.Stance.copy()

for target in targets:
    print 80 * "="
    print target
    print 80 * "="

    target_train_data = train_data[train_data.Target == target]
    true_stances = target_train_data.Stance
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

    vot_clf.fit(target_train_data.Tweet, true_stances)

    # predict on test data
    index = test_data.Target == target
    test_tweets = test_data.loc[index, 'Tweet']
    test_data.loc[index, 'Stance'] = vot_clf.predict(test_tweets)

    # predict on training data too to gauge overfitting
    index = train_data.Target == target
    train_tweets = train_data.loc[index, 'Tweet']
    pred_stances = vot_clf.predict(train_tweets)

    print classification_report(true_stances, pred_stances,
                            digits=4)

    macro_f = fbeta_score(true_stances, pred_stances, 1.0,
                          labels=['AGAINST', 'FAVOR'], average='macro')

    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(
            macro_f)

    # update stances training data
    train_data.loc[index, 'Stance'] = pred_stances


print 80 * "="
print 'Overall'
print 80 * "="

print classification_report(true_train_stances, train_data.Stance,
                            digits=4)

macro_f = fbeta_score(true_train_stances, train_data.Stance, 1.0,
                      labels=['AGAINST', 'FAVOR'], average='macro')

print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(
        macro_f)


train_fname = 'run1_train.txt'
print 'Writing ', train_fname
train_data.to_csv(open(train_fname, 'w'), '\t', encoding='utf-8',
                  quoting=csv.QUOTE_NONE)

test_fname = 'run1_test.txt'
print 'Writing ', test_fname
test_data.to_csv(open(test_fname, 'w'), '\t', encoding='utf-8',
                 quoting=csv.QUOTE_NONE)
