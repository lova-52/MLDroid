import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def data_processing(data):
    
    # Shuffle the rows of the dataset
    data = data.sample(frac=1)

    # Perform one-hot encoding on the Package and Category columns
    data = pd.get_dummies(data, columns=['Package', 'Category'])

    # Replace missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('Class', axis=1)
    X = imputer.fit_transform(X)
    
    # Apply min-max normalization to the selected features
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X)

    return X_new