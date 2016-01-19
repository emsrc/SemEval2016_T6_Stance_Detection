import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedKFold, PredefinedSplit

# data

train_data = pd.read_csv(open('semeval2016-task6-trainingdata-utf-8.txt'), '\t',
                         encoding='utf8',
                         index_col=0)
targets = list(train_data.Target.unique())

for target in targets:
    print 80 * "="
    print target
    print 80 * "="

    target_idx = train_data.Target == target
    target_train_data = train_data[target_idx]
    target_true_stances = target_train_data.Stance


    print 'training instances:', len(train_data)
    print 'target training instances:', len(target_train_data)

    target_cv = StratifiedKFold(target_true_stances, n_folds=5, shuffle=True,
                                random_state=13)

    predef_test_fold = -np.ones(len(train_data), dtype='int')
    predef_test_fold[np.where(target_idx)] = target_cv.test_folds

    train_cv = PredefinedSplit(predef_test_fold)

    for train, test in train_cv:
        print len(train), len(test), len(train) + len(test)
        print train_data.Target.iloc[test]
