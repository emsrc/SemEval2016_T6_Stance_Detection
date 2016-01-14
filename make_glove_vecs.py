"""
This uses Glove word vectors (http://nlp.stanford.edu/projects/glove/)
trained on 2B Twitter tweets (http://nlp.stanford.edu/data/glove.twitter.27B.zip)
to build vector representations for Climate Change tweets.
It simply collects the Glove vectors for all words in a tweet and
sums them.
"""

from codecs import open
from cStringIO import StringIO

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.preprocessing import normalize


data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t',
                   index_col=0)
target_data = data[data.Target == 'Climate Change is a Real Concern']

# First establish the vocabulary of the Climate Change tweets.
# lowercase because Glove terms are lowercased
# remove stopword because their vectors are probably not very meaningful
vectorizer = CountVectorizer(binary=True, lowercase=True, stop_words='english')
vectorizer.fit(target_data.Tweet)
tweet_vocab = set(vectorizer.get_feature_names())

# Local copies of Glove vectors of different dimensions
inf = '/Users/work/BigData/glove/glove.twitter.27B/glove.twitter.27B.{}d.txt'
outf = 'semeval2016-task6-trainingdata_climate_glove.twitter.27B.{}d.pkl'


for dim in 25, 50, 100, 200:
    # read Glove vectors
    # slurping the whole file with pd.read_cvs does not work as the table gets
    # get truncated! Presumably because of some kind of memory problem.
    # Hence the complicated approach below with a first pass through
    # the Glove file to collect the required vectors.
    vect_fname = inf.format(dim)
    buffer = StringIO()
    shared_vocab = []

    for line in open(vect_fname, encoding='utf8'):
        term = line.split(' ', 1)[0]
        if term in tweet_vocab:
            shared_vocab.append(term)
            buffer.write(line)

    print '#shared:', len(shared_vocab)
    buffer.seek(0)
    glove_vecs = pd.read_csv(buffer, sep=' ', header=None, index_col=0)
    buffer.close()

    # get Glove vectors as numpy.array
    glove_vecs = glove_vecs.as_matrix()
    #normalize(glove_vecs, copy=False)

    # vectorize our tweets with this shared vocabulary
    vectorizer = CountVectorizer(binary=True, stop_words='english',
                                 vocabulary=shared_vocab)
    tweet_vecs = vectorizer.fit_transform(target_data.Tweet)
    # convert sparse matrix to numpy.array (not needed?)
    tweet_vecs = np.squeeze(np.asarray(tweet_vecs.todense()))
    #normalize(tweet_vecs, copy=False)

    # take the dot product of the matrices,
    # which amounts to summing the Glove vectors for all terms in a tweet
    tweet_glove_vecs = tweet_vecs.dot(glove_vecs)

    tweet_glove_df = pd.DataFrame(tweet_glove_vecs,
                                  index=target_data.Tweet)
    tweet_glove_df.to_pickle(outf.format(dim))
