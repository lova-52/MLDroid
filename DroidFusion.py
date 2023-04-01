import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
dataset_file = 'D:\\uit\\bao mat web\\project mldroid\\AndroidPermission.csv'
data = pd.read_csv(dataset_file)

# Perform one-hot encoding on categorical columns
data = pd.get_dummies(data, columns=['App'])
data = pd.get_dummies(data, columns=['Package'])
data = pd.get_dummies(data, columns=['Category'])
data = pd.get_dummies(data, columns=['Description'])
data = pd.get_dummies(data, columns=['Related apps'])

# Perform feature selection using PCA
X = data.drop('Class', axis=1)
y = data['Class']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the batch size and number of components
batch_size = 10000
n_components = 30

# Create the Incremental PCA object
pca = IncrementalPCA(n_components=n_components)

# Apply Incremental PCA to the dataset in batches
for batch in np.array_split(X, len(X) // batch_size):
    pca.partial_fit(batch)

# Transform the data
X_new = pca.transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
model_file = 'D:\\uit\\bao mat web\\project mldroid\\rf_model.joblib'
joblib.dump(clf, model_file)
