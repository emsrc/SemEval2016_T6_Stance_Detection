import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score

from glove_transformer import GloveVectorizer

data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t',
                   index_col=0)
data = data[data.Target == 'Climate Change is a Real Concern']
true_stances = data.Stance

cv = StratifiedKFold(true_stances, n_folds=5, shuffle=True, random_state=1)


for dim in 25, 50, 100, 200:
    print 80 * '='
    print 'DIMENSIONS:', dim

    glove_fname = 'semeval2016-task6-trainingdata_climate_glove.twitter.27B.{}d.pkl'
    glove_vecs = pd.read_pickle(glove_fname.format(dim))

    pipeline = Pipeline([('vect', GloveVectorizer(glove_vecs)),
                         ('clf', SVC(C=1, gamma=0.01))])

    pred_stances = cross_val_predict(pipeline, data.Tweet, true_stances, cv=cv)
    print classification_report(true_stances, pred_stances, digits=4)

    macro_f = fbeta_score(true_stances, pred_stances, 1.0,
                          labels=['AGAINST', 'FAVOR'], average='macro')
    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)