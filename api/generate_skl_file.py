from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset (this assumes you have a CSV file)
df = pd.read_csv('heart.csv')

# Select the 13 relevant features
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']  # Target variable (heart disease or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a scaler and SVC model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(C=10, gamma='auto'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model to a pickle file
import pickle
with open('svm_13_features.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
