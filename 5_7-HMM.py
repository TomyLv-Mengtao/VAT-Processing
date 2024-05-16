import os
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Paths to positive and negative trace folders
pos_trace_path = "/workspaces/VAT-Processing/Duration_Slices/75/Pos"
neg_trace_path = "/workspaces/VAT-Processing/Duration_Slices/75/Neg"

# Mapping AOI names to integers
aoi_mapping = {"alt": 0, "att": 1, "spd": 2, "vsp": 3, "otw": 4, "nos": 5, "oth": 6}

# Function to load traces from a given directory
def load_traces(trace_path):
    traces = []
    for filename in os.listdir(trace_path):
        if filename.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(trace_path, filename), header=None, skiprows=1)
            # Ensure the first column is mapped and the second column is float
            df[0] = df[0].map(aoi_mapping)
            df[1] = pd.to_numeric(df[1], errors='coerce')
            df = df.dropna()  # Drop rows with NaN values
            traces.append(df.values)
    return traces

# Load positive and negative traces
pos_traces = load_traces(pos_trace_path)
neg_traces = load_traces(neg_trace_path)

# Combine positive and negative traces, and create labels
all_traces = pos_traces + neg_traces
labels = np.array([1] * len(pos_traces) + [0] * len(neg_traces))

# Convert traces to sequences for HMM
lengths = [len(trace) for trace in all_traces]
all_sequences = np.concatenate(all_traces)

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
precisions = []
recalls = []

for train_index, test_index in kf.split(all_traces):
    # Prepare training data
    train_sequences = np.concatenate([all_traces[i] for i in train_index])
    train_lengths = [len(all_traces[i]) for i in train_index]
    train_labels = labels[train_index]
    
    # Prepare test data
    test_sequences = np.concatenate([all_traces[i] for i in test_index])
    test_lengths = [len(all_traces[i]) for i in test_index]
    test_labels = labels[test_index]
    
    # Train HMM
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
    model.fit(train_sequences, lengths=train_lengths)
    
    # Predict labels for test data
    predicted_labels = []
    for test_trace in [all_traces[i] for i in test_index]:
        logprob, state_sequence = model.decode(test_trace)
        predicted_labels.append(state_sequence[0])
    
    # Calculate metrics
    accuracies.append(accuracy_score(test_labels, predicted_labels))
    precisions.append(precision_score(test_labels, predicted_labels))
    recalls.append(recall_score(test_labels, predicted_labels))

# Output average accuracy, precision, and recall
print(f"Average Accuracy: {np.mean(accuracies):.3f}")
print(f"Average Precision: {np.mean(precisions):.3f}")
print(f"Average Recall: {np.mean(recalls):.3f}")
