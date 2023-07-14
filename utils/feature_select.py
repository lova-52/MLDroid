from logging import info
from scikit_roughsets.rs_reduction import RoughSetsSelector

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_regression, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

import numpy as np
from itertools import chain, combinations
#Define gain ratio function
def gain_ratio(X, y):
    X = X.astype('int64')
    n_instances = len(X)
    n_classes = len(np.unique(y))

    # Calculate information entropy of class variable
    class_counts = np.bincount(y)
    class_probs = class_counts / n_instances
    I_X = -np.sum(class_probs * np.log2(class_probs))

    # Calculate gain ratio for each feature
    gain_ratios = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        feature_values = X[:, i]
        # Calculate information entropy of feature
        value_counts = np.bincount(feature_values)
        value_probs = value_counts / n_instances
        E_Z = 0
        for j in range(len(value_probs)):
            subset_indices = np.where(feature_values == j)[0]
            subset_class_counts = np.bincount(y[subset_indices], minlength=n_classes)
            subset_class_probs = subset_class_counts / len(subset_indices)
            subset_I_X = -np.sum(subset_class_probs * np.log2(subset_class_probs))
            E_Z += subset_I_X * (len(subset_indices) / n_instances)
        # Calculate split information of feature
        split_info = -np.sum(value_probs * np.log2(value_probs))
        # Calculate information gain of feature
        information_gain = I_X - E_Z
        # Calculate gain ratio of feature
        if split_info == 0:
            gain_ratios[i] = 0
        else:
            gain_ratios[i] = information_gain / split_info
    return gain_ratios

#Define OneR function
def OneR(X, y, k=4):
    # Get the number of samples and features
    n_samples, n_features = X.shape
    
    # Get the unique values in the feature matrix
    possible_thresholds = np.unique(X)
    
    # Initialize the list of selected features and best accuracy
    selected_features = []
    best_accuracy = 0
    
    # Loop over the desired number of selected features
    for _ in range(k):
        # Initialize the best feature and threshold for this iteration
        best_feature = None
        best_threshold = None
        
        # Loop over each feature
        for i in range(n_features):
            # Skip if the feature has already been selected
            if i in selected_features:
                continue
                
            # Get the feature values for this feature
            feature_values = X[:, i]
            
            # Loop over each unique value as a possible threshold
            for threshold in possible_thresholds:
                # Split the samples into left and right based on threshold
                left_idxs = feature_values <= threshold
                right_idxs = feature_values > threshold
                left_labels = y[left_idxs]
                right_labels = y[right_idxs]
                
                # Skip if either left or right set is empty
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue
                
                # Calculate the counts of each label in the left and right sets
                left_counts = np.bincount(left_labels)
                right_counts = np.bincount(right_labels)
                
                # Get the most frequent label in the left and right sets
                left_prediction = np.argmax(left_counts)
                right_prediction = np.argmax(right_counts)
                
                # Calculate the accuracy of this split
                accuracy = (left_counts[left_prediction] + right_counts[right_prediction]) / n_samples
                
                # Update the best feature and threshold if accuracy is higher
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = i
                    best_threshold = threshold
                    
        # Add the best feature to the selected features list
        if best_feature is not None:
            selected_features.append(best_feature)
        else:
            break
                
    return selected_features

# Define Univariate Logistic Regression ULR function
def ULR(X, y):
    clf = LogisticRegression(penalty='l2', solver='liblinear') #Default values
    clf.fit(X, y)
    coefs = clf.coef_[0]
    pvals = clf.predict_proba(X)[:, 1]
    features = list(range(X.shape[1]))
    coef_pval_vals = list(zip(features, coefs, pvals))
    coef_pval_vals = [x for x in coef_pval_vals if x[1] > 0 and x[2] < 0.05]
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

def filtered_subset_evaluation(X, y):
    # Apply an arbitrary filtering approach
    selector = VarianceThreshold(threshold=0.1)
    X_filtered = selector.fit_transform(X)
    
    # Train a logistic regression model and evaluate consistency
    clf = LogisticRegression()
    clf.fit(X_filtered, y)
    y_pred = clf.predict(X_filtered)
    consistency_rate = sum(y_pred == y) / len(y)
    
    # Select the subset of features with high consistency rate
    selector = SelectFromModel(clf, prefit=True, threshold=consistency_rate)
    X_new = selector.transform(X_filtered)
    
    return X_new

def rough_set_analysis(X, y):
    # Convert input features X to binary matrix
    binary_X = np.where(X > 0, 1, 0)
    print(binary_X)

    # Define function to calculate Information System
    def calculate_information_system_A(binary_X):
        features = binary_X.shape[1]
        combinations_list = combinations(range(features), r=features+1)
        information_system = {}
        for combination in combinations_list:
            rows = np.all(binary_X[:, combination], axis=1)
            classification = y[rows]
            information_system[tuple(combination)] = classification
        return information_system

    # Define function to calculate upper and lower approximations
    def calculate_approximations_A(information_system, binary_X):
        approximations = {}
        for feature_set, classification in information_system.items():
            rows = np.all(binary_X[:, feature_set], axis=1)
            upper_approximation = y[np.all(y[rows] == y[:, None], axis=0)]
            lower_approximation = y[np.any(y[rows] == y[:, None], axis=0)]
            approximations[feature_set] = (upper_approximation, lower_approximation)
        return approximations

    # Define function to calculate reduced set of features
    def calculate_reduced_attributes_A(approximations):
        reduced_attributes = []
        for feature_set, (upper, lower) in approximations.items():
            if np.array_equal(upper, lower):
                reduced_attributes.append(feature_set)
        return reduced_attributes

    # Calculate Information System
    information_system = calculate_information_system_A(binary_X)
    print(information_system)
    # Calculate upper and lower approximations
    approximations = calculate_approximations_A(information_system, binary_X)

    # Calculate reduced set of features
    reduced_features = calculate_reduced_attributes_A(approximations)

    # Return reduced set of features
    return reduced_features
#Define feature selecting function
def feature_selecting(feature_selection_name, X, y):
    try:
        if feature_selection_name == "FR1":
            selector = SelectKBest(chi2, k = 4)
            X_new = selector.fit_transform(X, y)
            return X_new
        elif feature_selection_name == "FR2":
            selector = SelectKBest(gain_ratio, k = 4)
            X_new = selector.fit_transform(X, y)
            return X_new
        elif feature_selection_name == "FR3":
            selected_feature_indices = OneR(X, y, k=4)
            X_new = X[:, selected_feature_indices]
            return X_new
        elif feature_selection_name == "FR4":
            selector = SelectKBest(mutual_info_classif, k = 4)
            X_new = selector.fit_transform(X, y)
            return X_new
        elif feature_selection_name == "FR5":
            X_new = ULR(X, y)
            return X_new
        elif feature_selection_name == "FR6":
            pca = PCA(n_components=4)
            X_new = pca.fit_transform(X)
            return X_new
        elif feature_selection_name == "FS1":
            selector = SelectKBest(score_func=f_regression, k=4)
            X_new = selector.fit_transform(X, y)
            return X_new
        elif feature_selection_name == "FS2":
            ICNR = consistency_subset_evaluation(X, y)  
            top_k_features = np.argsort(ICNR)[:4]
            X_new = X[:, top_k_features]
            return X_new
        elif feature_selection_name == "FS3":
            X_new = consistency_subset_evaluation(X, y)
            return X_new
        elif feature_selection_name == "FS4":            
            reduced_features = rough_set_analysis(X, y)
            X_new = X[:, reduced_features]
            return X
        else:
            raise ValueError(f"Invalid feature selection method: {feature_selection_name}")
    except ValueError as e:
        print(e)
        return None



