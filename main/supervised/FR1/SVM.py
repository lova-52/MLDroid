import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
from sklearn.metrics import f1_score, confusion_matrix
warnings.filterwarnings("ignore")

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
confusion_matrices = []

for dataset_file in dataset_files:

    # Load dataset
    data = pd.read_excel(f'D:\\uit\\BaoMatWeb\\MLDroid\\DATASET\\{dataset_file}')

    # Shuffle the rows of the dataset
    data = data.sample(frac=1)

    # Perform one-hot encoding on the Package and Category columns
    data = pd.get_dummies(data, columns=['Package', 'Category'])

    # Replace missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('Class', axis=1)
    X = imputer.fit_transform(X)

    # Perform feature selection using gain ratio
    y = data['Class']
    selector = SelectKBest(chi2, k=20)
    X_new = selector.fit_transform(X, y)

    # Apply min-max normalization to the selected features
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)

    # Train a SVM classifier using 20-fold cross-validation
    clf = SVC(kernel='poly', C=20, random_state=None, max_iter=1000)
    sk_folds = StratifiedKFold(n_splits=20, shuffle=True ,random_state = None)
    
    for train_index, test_index in sk_folds.split(X_new, y):
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
    
        accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)
    
        f_measure = f1_score(y_test, y_pred, average='weighted')
        f_measures.append(f_measure)

    accuracy_mean = np.mean(accuracies)
    f_measure_mean = np.mean(f_measures)
    
    # Print mean accuracy and f1 score
    print(f"{dataset_file}: Accuracy: {accuracy_mean:.4f}  F-measure: {f_measure_mean:.2f}")

    #Result:
    #D1_DATASET.xlsx: Accuracy: 0.8241  F-measure: 0.7868
    #D2_DATASET.xlsx: Accuracy: 0.8801  F-measure: 0.8539
    #D3_DATASET.xlsx: Accuracy: 0.8773  F-measure: 0.8480
    #D4_DATASET.xlsx: Accuracy: 0.8834  F-measure: 0.8561
    #D5_DATASET.xlsx: Accuracy: 0.8805  F-measure: 0.8550
    #D6_DATASET.xlsx: Accuracy: 0.8848  F-measure: 0.8592
    #D7_DATASET.xlsx: Accuracy: 0.8940  F-measure: 0.8702
    #D8_DATASET.xlsx: Accuracy: 0.8823  F-measure: 0.8590
    #D9_DATASET.xlsx: Accuracy: 0.8850  F-measure: 0.8618
    #D10_DATASET.xlsx: Accuracy: 0.8879  F-measure: 0.8644
    #D11_DATASET.xlsx: Accuracy: 0.8901  F-measure: 0.8670
    #D12_DATASET.xlsx: Accuracy: 0.8891  F-measure: 0.8649
    #D13_DATASET.xlsx: Accuracy: 0.8965  F-measure: 0.8736
    #D14_DATASET.xlsx: Accuracy: 0.8994  F-measure: 0.8772
    #D15_DATASET.xlsx: Accuracy: 0.8994  F-measure: 0.8771
    #D16_DATASET.xlsx: Accuracy: 0.9042  F-measure: 0.8829
    #D17_DATASET.xlsx: Accuracy: 0.9033  F-measure: 0.8818
    #D18_DATASET.xlsx: Accuracy: 0.9054  F-measure: 0.8844
    #D19_DATASET.xlsx: Accuracy: 0.9063  F-measure: 0.8852
    #D20_DATASET.xlsx: Accuracy: 0.9071  F-measure: 0.8864
    #D21_DATASET.xlsx: Accuracy: 0.9027  F-measure: 0.8816
    #D22_DATASET.xlsx: Accuracy: 0.9026  F-measure: 0.8817
    #D23_DATASET.xlsx: Accuracy: 0.9030  F-measure: 0.8823
    #D24_DATASET.xlsx: Accuracy: 0.9006  F-measure: 0.8790
    #D25_DATASET.xlsx: Accuracy: 0.9027  F-measure: 0.8816
    #D26_DATASET.xlsx: Accuracy: 0.9033  F-measure: 0.8826
    #D27_DATASET.xlsx: Accuracy: 0.8958  F-measure: 0.8757
    #D28_DATASET.xlsx: Accuracy: 0.8977  F-measure: 0.8780
    #D29_DATASET.xlsx: Accuracy: 0.8967  F-measure: 0.8761
    #D30_DATASET.xlsx: Accuracy: 0.8975  F-measure: 0.8772