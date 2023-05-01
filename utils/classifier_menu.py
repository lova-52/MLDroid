import sys
from utils.model_train import *

def print_menu():
    print("Select a classifier:")
    print("1. SVM")
    print("2. Naive Bayes")
    print("3. Random Forest")
    print("4. MLP")
    print("5. Logistic Regression")
    print("6. Bayesian Network")
    print("7. AdaBoost")
    print("8. Decision Tree")
    print("9. k-Nearest Neighbors")
    print("10. Deep Neural Network")
    print("0. Exit")

    classifier_choice = input("Enter your choice: ")

    if classifier_choice == "0":
        sys.exit()

    print("Select a feature selection method:")
    print("1. FR1")
    print("2. FR2")
    print("3. FR3")
    print("4. FR4")
    print("5. FR5")
    print("6. FR6")

    feature_selection_choice = input("Enter your choice: ")

    return classifier_choice, feature_selection_choice

# Main program loop
while True:
    try:
        classifier_choice, feature_selection_choice = print_menu()

        # Convert choices to classifier and feature selection method names
        classifiers = ["SVM", "NB", "RF", "MLP", "LR", "BN", "AB", "DT", "KNN", "DNN"]
        classifier_name = classifiers[int(classifier_choice) - 1]

        feature_selections = ["FR1", "FR2", "FR3", "FR4", "FR5", "FR6"]
        feature_selection_name = feature_selections[int(feature_selection_choice) - 1]

        # Train model
        model_training(classifier_name, feature_selection_name)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit()

