from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

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

#Define Informain-gain function
def information_gain(X, y):
    n_features = X.shape[1]
    X = X.astype('int64')
    # Calculate information entropy of class variable
    class_counts = np.bincount(y)
    class_probs = class_counts / len(y)
    entropy = -np.sum(class_probs * np.log2(class_probs))

    # Calculate information gain for each feature
    information_gains = np.zeros(n_features)
    for i in range(n_features):
        feature_values = X[:, i]
        # Calculate information entropy of feature
        value_counts = np.bincount(feature_values)
        value_probs = value_counts / len(feature_values)
        value_entropy = -np.sum(value_probs * np.log2(value_probs))
        # Calculate information gain of feature
        value_class_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(class_counts)), 0, [feature_values, y])
        value_class_probs = value_class_counts / len(y)
        value_entropy_class = -np.sum(value_class_probs * np.log2(value_class_probs), axis=1)
        value_entropy_class_weighted = np.sum(value_probs * value_entropy_class)
        information_gain = entropy - value_entropy_class_weighted
        # Add information gain of feature
        information_gains[i] = information_gain

    # Sort features by information gain
    feature_indices = np.argsort(information_gains)[::-1]
    return feature_indices

# Define Univariate Logistic Regression ULR function
def ULR(X, y):
    clf = LogisticRegression(penalty='l2', solver='liblinear') #Default values
    clf.fit(X, y)
    coefs = clf.coef_[0]
    pvals = clf.predict_proba(X)[:, 1]
    features = list(range(X.shape[1]))
    coef_pval_vals = list(zip(features, coefs, pvals))
    coef_pval_vals.sort(key=lambda x: abs(x[1]), reverse=True)
    return coef_pval_vals

#Define Consistency subset evaluation function
def consistency_subset_evaluation(X, y):
    Z = X.shape[1]      #Z = number of features
    ICNR = np.zeros(Z)  #ICNR = the inconsistency rate for each feature
    for i in range(Z): 
        A = X[:, i]     #A = the values of a single feature across all samples
        INC = 0         #INC = the count of inconsistent samples for a single feature
        Z_i = 0         #Z_i = the number of samples with the current feature value
        for j in range(len(A)):
            #Count the number of samples with the same feature value but different class label
            A0 = np.sum(np.logical_and(A == A[j], y == 0))
            A1 = np.sum(np.logical_and(A == A[j], y == 1))
            if A1 < A0:
                INC += A0 - A1
            Z_i += 1
        #Calculate inconsistency rate for the feature
        ICNR[i] = INC / Z_i
    return ICNR

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
    elif feature_selection_name == "FR4":
        feature_indices = information_gain(X, y)
        X_new = X[:, feature_indices[:20]]
        return X_new
    elif feature_selection_name == "FR5":
        ranked_features = ULR(X, y)
        k = 20 #Select top 20 features
        top_k_features = [x[0] for x in ranked_features[:k]]
        X_new = X[:, top_k_features]
        return X_new
    elif feature_selection_name == "FR6":
        pca = PCA(n_components=20)
        X_new = pca.fit_transform(X)
        return X_new
    elif feature_selection_name == "FS1":
        selector = SelectKBest(score_func=f_regression, k=20)
        X_new = selector.fit_transform(X, y)
        return X_new
    elif feature_selection_name == "FS2":
        ICNR = consistency_subset_evaluation(X, y)
        #Select top 20 features based on inconsistency rates
        top_k_features = np.argsort(ICNR)[:20]
        X_new = X[:, top_k_features]
        return X_new
    else:
        raise ValueError(f"Invalid feature selection method: {feature_selection_name}")


