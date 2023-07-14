import numpy as np
import random
import re
import warnings
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import (
    BernoulliNB as BayesianClassifer,
    GaussianNB,
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from tabulate import tabulate

from utils.data_process import data_processing, min_max_normalize
from utils.dataset_loader import *
from utils.feature_select import feature_selecting

import seaborn as sns
import matplotlib.pyplot as plt

#Ignore warnings
warnings.filterwarnings("ignore")

def choose_classifier(classifier_name):
    if classifier_name == "SVM":
        # Train a SVM classifier using 20-fold cross-validation
        clf = SVC(kernel='poly', C=69, random_state=None, max_iter=1000)
        pass
    elif classifier_name == "NB":
        # Train a Naive Bayes classifier using 20-fold cross-validation
        clf = GaussianNB()
        pass
    elif classifier_name == "RF":
        # Train a Random Forest classifier using 20-fold cross-validation
        clf = RandomForestClassifier(n_estimators=100, random_state=None)
        pass
    elif classifier_name == "MLP":
        # Train a MLP classifier using 20-fold cross-validation
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=None)
        pass
    elif classifier_name == "LR":
        # Train a Logistic Regression classifier using 20-fold cross-validation
        clf = clf = LogisticRegression(random_state=None)
        pass
    elif classifier_name == "BN":
        # Train a Bayesian Network classifier using 20-fold cross-validation
        clf = BayesianClassifer()
        pass
    elif classifier_name == "AB":
        # Train an AdaBoost classifier using 20-fold cross-validation
        clf = AdaBoostClassifier(n_estimators=100, random_state=None)
        pass
    elif classifier_name == "DT":
        # Train a Decision Tree classifier using 20-fold cross-validation
        clf = DecisionTreeClassifier(random_state=None)
        pass
    elif classifier_name == "KNN":
        # Train a k-Nearest Neighbors classifier using 20-fold cross-validation
        clf = KNeighborsClassifier(weights=random.choice(['uniform', 'distance']))
        pass
    elif classifier_name == "DNN":
        # Train a Deep Neural Network classifier using 20-fold cross-validation
        clf = MLPClassifier(hidden_layer_sizes=(200, 150, 100, 50), max_iter=1000)
        pass
    else:
        print("Invalid classifier name.")
    return clf

def print_results(results):
    #Sort results
    results.pop(0)
    results_sorted = sorted(results, key=lambda x: int(re.findall(r'\d+', x[0])[0]))

    #Make a list of headers for the table
    headers = ["Dataset", "Accuracy", "F-measure"]

    #Make a list of rows for the table
    rows = []
    for result in results_sorted:
        rows.append([result[0], f"{result[1]*100:.2f}", f"{result[2]:.2f}"])

    #Print table
    print(tabulate(rows, headers=headers))

def model_training(classifier_name, feature_selection_name):

    # List of dataset filenames
    dataset_files = assign_dataset()

    # Dictionary to hold the sum of confusion matrices for each dataset file
    confusion_matrices_sum = {}

    # List to hold confusion matrices, accuracies, f-measures, and results
    confusion_matrices = []
    accuracies = []
    f_measures = []
    results = []

    # Count the loaded datasets
    count = 0

    print("Training", classifier_name, "classifier...")

    while True:
        for dataset_file in dataset_files:
            # Load the dataset           
            data = load_dataset(dataset_file)

            # Process data
            y = data['Class']
            X_new = data_processing(data)

            # Perform feature selection
            X_new = feature_selecting(feature_selection_name, X_new, y)

            # Apply min-max normalization to the selected features
            X_new = min_max_normalize(X_new)

            # Set the classifier
            clf = choose_classifier(classifier_name)
            
            # Cross-validation with k = 20
            sk_folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=None)

            # Train the model
            for train_index, test_index in sk_folds.split(X_new, y):
                X_train, X_test = X_new[train_index], X_new[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Calculate and store the confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                confusion_matrices.append(cm)

                accuracy = clf.score(X_test, y_test)
                accuracies.append(accuracy)

                f_measure = f1_score(y_test, y_pred, average='weighted')
                f_measures.append(f_measure)

            accuracy_mean = np.mean(accuracies)
            f_measure_mean = np.mean(f_measures)

            # Save the results to print them later
            results.append((dataset_file, accuracy_mean, f_measure_mean))
            
         
            count += 1
            if count == 31:
                break

            # Sum the confusion matrices for the current dataset file
            confusion_matrices_sum[dataset_file] = np.sum(confusion_matrices, axis=0)

        if count == 31:
            break   

    # Print the results
    print_results(results)

    # Plot the confusion matrices
    for dataset_file, cm_sum in confusion_matrices_sum.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_sum, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
