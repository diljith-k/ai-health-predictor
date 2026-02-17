import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Assuming first column is Disease
y = df.iloc[:, 0]

# Remaining columns contain symptoms (text)
X_raw = df.iloc[:, 1:]

# Convert text symptoms into dummy variables (One-Hot Encoding)
X = pd.get_dummies(X_raw)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save feature names (IMPORTANT for Flask app)
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))

print("Model trained successfully!")
print("Number of features:", model.n_features_in_)

