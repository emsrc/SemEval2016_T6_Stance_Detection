#!/usr/bin/env python

"""
This uses Glove word vectors (http://nlp.stanford.edu/projects/glove/)
trained on 2B Twitter tweets (http://nlp.stanford.edu/data/glove.twitter.27B.zip)
to build vector representations all tweets.
It simply collects the Glove vectors for all words in a tweet and
sums them.

Requires python2 with Numpy, Pandas en sklearn
"""



from codecs import open
from cStringIO import StringIO
from glob import glob
from os.path import basename, splitext

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t')
test_data = pd.read_csv(open('SemEval2016-Task6-subtaskA-testdata.txt'), '\t')
data = pd.concat((train_data, test_data))

# First establish the vocabulary of all tweets.
# lowercase because Glove terms are lowercased
# remove stopword because their vectors are probably not very meaningful
vectorizer = CountVectorizer(binary=True, lowercase=True, stop_words='english',
                             decode_error='ignore')
vectorizer.fit(data.Tweet)
tweet_vocab = set(vectorizer.get_feature_names())

# base dir for local copies of Glove vectors for different corpora & dimensions
base_dir = '/Users/work/BigData/glove'

# glob pattern for Glove vectors
glove_fnames = glob(base_dir + '/*.txt')# + glob(base_dir + '/*/*.txt')

out_dir = 'glove_vecs'

# Read the Glove vectors Slurping the whole file with pd.read_cvs does not
# work as the table gets get truncated! Presumably because of some kind of
# memory problem. Hence the complicated approach below with a first pass
# through the Glove file to collect the required vectors in an buffer.

for fname in glove_fnames:
    print 'reading', fname
    buffer = StringIO()
    shared_vocab = []

    for line in open(fname, encoding='utf8'):
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

    # vectorize our tweets with this shared vocabulary
    vectorizer = CountVectorizer(binary=True, stop_words='english',
                                 vocabulary=shared_vocab, decode_error='ignore')
    tweet_vecs = vectorizer.fit_transform(data.Tweet)
    # convert sparse matrix to numpy.array (not needed?)
    tweet_vecs = np.squeeze(np.asarray(tweet_vecs.todense()))

    # take the dot product of the matrices,
    # which amounts to summing the Glove vectors for all terms in a tweet
    tweet_glove_vecs = tweet_vecs.dot(glove_vecs)

    # save vector as DataFrame with tweets as index
    tweet_glove_df = pd.DataFrame(tweet_glove_vecs, index=data.Tweet)
    out_fname = out_dir + '/' + splitext(basename(fname))[0] + '_semeval2016-task6.pkl'
    print 'writing', out_fname
    tweet_glove_df.to_pickle(out_fname)
