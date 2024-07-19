import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=column_names)

# Replace missing values marked with '?' with NaN and drop those rows
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Define features and target
X = data.drop("target", axis=1)
y = data["target"].apply(lambda x: 1 if x > 0 else 0)  # Convert to binary classification

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = SVC(probability=True)
model.fit(X_train, y_train)

# Save the model as a .pkl file
with open("svc.pkl", "wb") as file:
    pickle.dump(model, file)

