"""
sample how to use FunctionTransformer and FeatureUnion in Pipeline
"""

import pandas as pd
from scipy import sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report


data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t',
                   index_col=0)
target_data = data[data.Target == 'Climate Change is a Real Concern']

cv = StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True,
                     random_state=1)


def has_negation(raw_docs):
    """
    feature indicating if document contains the negation 'not'
    """
    v = ['not' in doc for doc in raw_docs]
    # Alternatively, as raw docs is a pandas.Series, this can be done faster as
    # v = raw_docs.str.contains('not')

    # Convert to 1/0 sparse matrix, transpose to get right dimensions
    # because this needs to ne concatenated with the sparse matrix output
    # from CountVectorizer.
    return sparse.csr_matrix(v, dtype='int').T


# Construct a pipeline that joins the features from two feature extractors and
# passes the result to a classifier
# (need validate=False because the input is not a matrix but raw text)

pipeline = make_pipeline(
    make_union(
        CountVectorizer(),
        FunctionTransformer(has_negation, validate=False)),
    MultinomialNB())

# More verbose, with explicit naming of each step:
#
# pipeline = Pipeline([
#     ('feat_extract', FeatureUnion([
#         ('vect', CountVectorizer()),
#         ('neg', FunctionTransformer(has_negation, validate=False))])),
#     ('clf', MultinomialNB())])


pred_stances = cross_val_predict(pipeline, target_data.Tweet,
                                 target_data.Stance, cv=cv)
print classification_report(target_data.Stance, pred_stances, digits=4)

macro_f = fbeta_score(target_data.Stance, pred_stances, 1.0,
                      labels=['AGAINST', 'FAVOR'], average='macro')
print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(
        macro_f)
