import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import joblib
import os

# List of dataset filenames
dataset_files = ['D1_DATASET.xlsx',
                 'D2_DATASET.xlsx',
                 'D3_DATASET.xlsx',
                 'D4_DATASET.xlsx',
                 'D5_DATASET.xlsx',
                 'D6_DATASET.xlsx',
                 'D7_DATASET.xlsx',
                 'D8_DATASET.xlsx',
                 'D9_DATASET.xlsx',
                 'D10_DATASET.xlsx',
                 'D11_DATASET.xlsx',
                 'D12_DATASET.xlsx',
                 'D13_DATASET.xlsx',
                 'D14_DATASET.xlsx',
                 'D15_DATASET.xlsx',
                 'D16_DATASET.xlsx',
                 'D17_DATASET.xlsx',
                 'D18_DATASET.xlsx',
                 'D19_DATASET.xlsx',
                 'D20_DATASET.xlsx',
                 'D21_DATASET.xlsx',
                 'D22_DATASET.xlsx',
                 'D23_DATASET.xlsx',
                 'D24_DATASET.xlsx',
                 'D25_DATASET.xlsx',
                 'D26_DATASET.xlsx',
                 'D27_DATASET.xlsx',
                 'D28_DATASET.xlsx',
                 'D29_DATASET.xlsx',
                 'D30_DATASET.xlsx']

# Lists to hold accuracy and f1 score values for each dataset
accuracies = []
f_measures = []

for dataset_file in dataset_files:

    # Load dataset
    data = pd.read_excel(os.path.join('D:\\uit\\bao mat web\\project mldroid\\DATASET', dataset_file))

    # Automatically add new columns
    for column in data.columns:
        if column not in ['Package', 'Category', 'Class']:
            data[f'{column}_missing'] = data[column].isnull().astype(int)

    # Perform one-hot encoding on the Package and Category columns
    data = pd.get_dummies(data, columns=['Package', 'Category'])

    # Replace missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('Class', axis=1)
    X = imputer.fit_transform(X)

    # Perform feature selection using chi-squared test with k = k = log2(1400)~~11
    y = data['Class']
    selector = SelectKBest(chi2, k=11)
    X_new = selector.fit_transform(X, y)

    # Get the boolean mask of selected features
    mask = selector.get_support()

    # Select the columns corresponding to the selected features
    selected_columns = X.columns[mask]

    # Create a DataFrame with only the selected features
    X_new = pd.DataFrame(X_new, columns=selected_columns)

    # Apply min-max normalization to the selected features
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)

    # Train a Bayesian network using 20-fold cross-validation
    X_new = pd.DataFrame(X_new)
    model = BayesianModel([('Category_Class', 'Package_Class'), ('Category_Class', 'Permission_Class'), ('Category_Class', 'Call_Class'), ('Category_Class', 'Activity_Class'), ('Category_Class', 'Service_Class'), ('Category_Class', 'Receiver_Class')])
    estimator = BayesianEstimator(model, X_new)
    scores = cross_val_score(estimator, X_new, y, cv=20)
    accuracy = scores.mean()
    accuracies.append(accuracy)
    print(f"{dataset_file}: {accuracy}")

    # Fit the model on the full dataset and save it
    model.fit(X_new, y)
    model_file = os.path.join('D:\\uit\\bao mat web\\project mldroid\\Trained Models', f'{dataset_file.replace(".xlsx", "_BayesianNetwork_model.joblib")}')
    joblib.dump(model, model_file)

    # Make predictions on the full dataset and calculate f1 score
    f_measure = f1_score(y, model.predict(X_new), average='weighted')
    f_measures.append(f_measure)
    print(f"{dataset_file}: {f_measure}")
