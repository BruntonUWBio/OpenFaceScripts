import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier


# Score on the training set was:0.9993190328689782
def happy_pipeline():
    return make_pipeline(
        StackingEstimator(
            estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=1.0, min_samples_leaf=1,
                                           min_samples_split=3, n_estimators=100)),
        StandardScaler(),
        SelectFwe(score_func=f_classif, alpha=0.026000000000000002),
        StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=2, p=1, weights="distance")),
        XGBClassifier(learning_rate=0.001, max_depth=6, min_child_weight=12, n_estimators=100, nthread=1,
                      subsample=0.9500000000000001)
    )


def angry_pipeline():
    # Score on the training set was:1.0
    return RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.9000000000000001,
                                  min_samples_leaf=1, min_samples_split=20, n_estimators=100)


def disgust_pipeline():
    # Score on the training set was:1.0
    return LinearSVC(C=0.1, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)


def fear_pipeline():
    # Score on the training set was:1.0
    return XGBClassifier(learning_rate=0.001, max_depth=9, min_child_weight=2, n_estimators=100, nthread=1,
                         subsample=0.3)


def sad_pipeline():
    # Score on the training set was:1.0
    return KNeighborsClassifier(n_neighbors=11, p=1, weights="distance")


def surprise_pipeline():
    # Score on the training set was:1.0
    return ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35000000000000003,
                                min_samples_leaf=5, min_samples_split=15, n_estimators=100)


if __name__ == '__main__':
    # NOTE: Make sure that the class is labeled 'target' in the data file
    tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
    features = tpot_data.drop('target', axis=1).values
    training_features, testing_features, training_target, testing_target = \
        train_test_split(features, tpot_data['target'].values, random_state=42)

    exported_pipeline = happy_pipeline()
    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
