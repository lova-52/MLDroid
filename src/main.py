from utils.classifier_menu import *


# Main program loop
while True:
    try:
        classifier_choice, feature_selection_choice = print_menu()

        # Convert choices to classifier and feature selection method names
        classifiers = ["SVM", "NB", "RF", "MLP", "LR", "BN", "AB", "DT", "KNN", "DNN"]
        classifier_name = classifiers[int(classifier_choice) - 1]

        feature_selections = ["FR1", "FR2", "FR3", "FR4", "FR5", "FR6", "FS1", "FS2", "FS3", "FS4"]
        feature_selection_name = feature_selections[int(feature_selection_choice) - 1]

        # Train model
        model_training(classifier_name, feature_selection_name)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit()