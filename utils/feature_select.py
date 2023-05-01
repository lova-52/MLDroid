from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold
import numpy as np

#Define gain ratio function
def gain_ratio(X, y):
    n_features = X.shape[1]
    X = X.astype('int64')
    # Calculate information entropy of class variable
    class_counts = np.bincount(y)
    class_probs = class_counts / len(y)
    entropy = -np.sum(class_probs * np.log2(class_probs))

    # Calculate gain ratio for each feature
    gain_ratios = np.zeros(n_features)
    for i in range(n_features):
        feature_values = X[:, i]
        # Calculate information entropy of feature
        value_counts = np.bincount(feature_values)
        value_probs = value_counts / len(feature_values)
        value_entropy = -np.sum(value_probs * np.log2(value_probs))
        # Calculate split information of feature
        split_info = -np.sum(value_probs * np.log2(value_probs))
        # Calculate information gain of feature
        value_class_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(class_counts)), 0, [feature_values, y])
        value_class_probs = value_class_counts / len(y)
        value_entropy_class = -np.sum(value_class_probs * np.log2(value_class_probs), axis=1)
        value_entropy_class_weighted = np.sum(value_probs * value_entropy_class)
        information_gain = entropy - value_entropy_class_weighted
        # Calculate gain ratio of feature
        if split_info == 0:
            gain_ratios[i] = 0
        else:
            gain_ratios[i] = information_gain / split_info
    return gain_ratios

#Define feature selecting function
def feature_selecting(feature_selection_name, X, y):
    if feature_selection_name == "FR1":
        selector = SelectKBest(chi2, k = 20)
        X_new = selector.fit_transform(X, y)
        return X_new
    elif feature_selection_name == "FR2":
        selector = SelectKBest(gain_ratio, k=20)
        X_new = selector.fit_transform(X, y)
        return X_new
    elif feature_selection_name == "FR3":
        selector = VarianceThreshold(threshold=0.1)
        X_new = selector.fit_transform(X)
        return X_new

