from pandas import read_csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, cross_val_score, \
    StratifiedKFold
from sklearn.metrics import fbeta_score

data = read_csv(open('semeval2016-task6-trainingdata.txt'), '\t',
                index_col=0)
targets = list(data.Target.unique()) + ['All']

pipeline = Pipeline([
    ('vect', CountVectorizer(decode_error='ignore',
                             #binary=True,
                             #ngram_range=(1,2)
                             )),
    #('clf', MultinomialNB())
    ('clf', LinearSVC(C=1.0))
])

for target in targets:
    print 80 * "="
    print target
    print 80 * "="

    if target == 'All':
        target_data = data
    else:
        target_data = data[data.Target == target]

    cv = StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True,
                         random_state=1)
    pred_stances = cross_val_predict(pipeline, target_data.Tweet,
                                     target_data.Stance, cv=cv)
    print classification_report(target_data.Stance, pred_stances, digits=4)

    macro_f = fbeta_score(target_data.Stance, pred_stances, 1.0,
                          labels=['AGAINST', 'FAVOR'], average='macro')
    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'. \
        format(macro_f)


# write to file to check with eval.pl
# data.to_csv(open('true.txt', 'w'), '\t')
# data.Stance = pred_stances
# data.to_csv(open('pred.txt', 'w'), '\t')

# TODO: why does cross_val_score give different scores!?
# def scorer(estimator, tweets, true_stances):
#     pred_stances = estimator.predict(tweets)
#     return fbeta_score(true_stances, pred_stances, 1.0,
#                        labels=['AGAINST', 'FAVOR'], average='macro')
#
#
# scores = cross_val_score(pipeline, data.Tweet, data.Stance, scoring=scorer,
#                          cv=cv)
# print scores.mean()
