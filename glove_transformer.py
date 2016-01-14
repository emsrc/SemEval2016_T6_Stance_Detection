from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

# For use in Pipeline, patch function 'clone' file sklearn/base.py
# by replacing
#
# equality_test = (new_obj_val == params_set_val or
#                 new_obj_val is params_set_val)
#
# with
#
# equality_test = True
#
# See https://github.com/scikit-learn/scikit-learn/issues/5522

class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, glove_vectors):
        self.glove_vectors = glove_vectors

    def fit(self, raw_documents, y=None):
        return self

    def fit_transform(self, raw_documents, y=None):
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        return self.glove_vectors.loc[raw_documents].as_matrix()


if __name__ == '__main__':
    import pandas as pd

    fname = 'semeval2016-task6-trainingdata_climate_glove.twitter.27B.25d.pkl'
    glove_vectors = pd.read_pickle(fname)

    data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t',
                       index_col=0)
    data = data[data.Target == 'Climate Change is a Real Concern']

    vectorizer = GloveVectorizer(glove_vectors)

    X = vectorizer.fit_transform(data.Tweet)
    print X.shape
