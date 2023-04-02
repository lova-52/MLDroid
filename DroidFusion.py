import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
dataset_file = 'D:\\uit\\bao mat web\\project mldroid\\DATASET_S_with_Class.xlsx'
data = pd.read_excel(dataset_file)

# Perform 5 one-hot encoding on the 5 columns
data = pd.get_dummies(data, columns=['Package'])
data = pd.get_dummies(data, columns=['Category'])


# Replace missing values with the mean
imputer = SimpleImputer(strategy='mean')
X = data.drop('Class', axis=1)
X = imputer.fit_transform(X)

# Perform feature selection using PCA
y = data['Class']
pca = PCA(n_components=50)
X_new = pca.fit_transform(X)

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
model_file = 'D:\\uit\\bao mat web\\project mldroid\\ft_all_mcl_rf.joblib'
joblib.dump(clf, model_file)
