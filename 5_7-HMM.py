# Step 5. method 7, Evaluate the performance of VAT with HMM
# 2024-05-16
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from hmmlearn import hmm

# Load data
pos_files = glob.glob("/workspaces/VAT-Processing/ML/Pos/10/*.csv")
neg_files = glob.glob("/workspaces/VAT-Processing/ML/Neg/10/*.csv")

pos_data = [pd.read_csv(file) for file in pos_files]
neg_data = [pd.read_csv(file) for file in neg_files]

# Concatenate all data into one DataFrame
pos_df = pd.concat(pos_data, ignore_index=True)
neg_df = pd.concat(neg_data, ignore_index=True)

# Add labels
pos_df['label'] = 1
neg_df['label'] = 0

# Combine positive and negative examples
data = pd.concat([pos_df, neg_df], ignore_index=True)

# Convert object columns to int, else will be object type error, 02-07
for col in ["duration", "dwell_alt", "dwell_att", "dwell_spd", "dwell_vsp", "dwell_otw", "dwell_nos", "dwell_oth"]:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

# Prepare features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Initialize the classifier
n_components = 2  # Number of states in the HMM
clf = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=10)

# Perform cross-validation and compute metrics
accuracy = []
precision = []
recall = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train HMM on positive and negative sequences separately
    clf_pos = clf.fit(X_train[y_train == 1])
    clf_neg = clf.fit(X_train[y_train == 0])
    
    # Predict based on log likelihood
    log_likelihood_pos = clf_pos.score_samples(X_test)
    log_likelihood_neg = clf_neg.score_samples(X_test)
    
    predictions = (log_likelihood_pos > log_likelihood_neg).astype(int)

    accuracy.append(accuracy_score(y_test, predictions))
    precision.append(precision_score(y_test, predictions))
    recall.append(recall_score(y_test, predictions))

# Print the results
print(f'Accuracy: {np.mean(accuracy):.3f}')
print(f'Precision: {np.mean(precision):.3f}')
print(f'Recall: {np.mean(recall):.3f}')
