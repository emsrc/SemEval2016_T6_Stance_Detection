from cPickle import dump
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer

from glove_transformer import GloveVectorizer

data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t',
                   index_col=0)

# When doing CV on only climate data, scores across folds vary widely (SD > 0.2)
# indicating insufficient training data.
#data = data[data.Target == 'Climate Change is a Real Concern']
true_stances = data.Stance

cv = StratifiedKFold(true_stances, n_folds=5, shuffle=True, random_state=7)

glove_fnames = ('glove_vecs/glove.42B.300d_semeval2016-task6.pkl',
                'glove_vecs/glove.6B.300d_semeval2016-task6.pkl',
                'glove_vecs/glove.840B.300d_semeval2016-task6.pkl',
                'glove_vecs/glove.twitter.27B.200d_semeval2016-task6.pkl'
                )

glove_ids = [fname.split('/')[-1].split('_')[0] for fname in glove_fnames]

general_params = dict(
        LR__class_weight=[None, 'balanced'],
        LR__C=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, ],
        # tol has no effect?
        #LR__tol=[1e-10, 1e-4],
        #LR__max_iter=[100, 250],

)

params = [
    dict(LR__solver=['liblinear'],
         # l2 works better on non-sparse
         #LR__penalty=['l1', 'l2'],
         #LR__intercept_scaling = [0.001, 0.01, 1, 10, 25, 50]
         ),
    dict(LR__solver=['lbfgs', 'newton-cg'],
         LR__multi_class=['ovr', 'multinomial'],
         #LR__max_iter=[100,250],
         #LR__warm_start=[True, False]
         ),
]

for p in params:
    p.update(general_params)


macro_f_scorer = make_scorer(fbeta_score,
                             beta=1.0,
                             labels=['AGAINST', 'FAVOR'],
                             average='macro')

results = {}

for fname, glove_id in zip(glove_fnames, glove_ids):  # [:1]:
    print 80 * '='
    print 'GLOVE VECTORS:', glove_id
    print 80 * '='

    glove_vecs = pd.read_pickle(fname)

    glove_clf = Pipeline([('vect', GloveVectorizer(glove_vecs)),
                          ('LR', LogisticRegression())])

    grid_search = GridSearchCV(glove_clf, params, scoring=macro_f_scorer, cv=cv)
    grid_search.fit(data.Tweet, true_stances)

    for s in sorted(grid_search.grid_scores_,
                    key=lambda x: x.mean_validation_score,
                    reverse=True)[:25]:
        print s, s.cv_validation_scores

    results[glove_id] = grid_search

dump(results, open('glove_grid_search.pkl', 'wb'))

