# Step 5. method 1, Evaluate the performance of VAT with LDA method
# 2024-02-07
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load data
pos_files = glob.glob("/workspaces/VAT-Processing/ML/Pos/75/*.csv")
neg_files = glob.glob("/workspaces/VAT-Processing/ML/Neg/75/*.csv")

# Read the files and assign labels
pos_data = [pd.read_csv(file).assign(target='pos') for file in pos_files]
neg_data = [pd.read_csv(file).assign(target='neg') for file in neg_files]

# Concatenate all data into one DataFrame
data = pd.concat(pos_data + neg_data, ignore_index=True)

# Ensure the data is in the correct format
data = data[["duration", "dwell_alt", "dwell_att", "dwell_spd", "dwell_vsp", "dwell_otw", "dwell_nos", "dwell_oth", "target"]]

# Convert labels into numerical values
le = LabelEncoder()
data['target'] = le.fit_transform(data['target'])

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Initialize the model
model = LDA()

# Perform 10-fold cross-validation
kf = KFold(n_splits=10)
accuracy = []
precision = []
recall = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy.append(accuracy_score(y_test, predictions))
    precision.append(precision_score(y_test, predictions))
    recall.append(recall_score(y_test, predictions))

# Print the results
print(f'Accuracy: {np.mean(accuracy):.3f}')
print(f'Precision: {np.mean(precision):.3f}')
print(f'Recall: {np.mean(recall):.3f}')
