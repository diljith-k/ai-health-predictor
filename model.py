import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Fill missing values with empty string
df.fillna("", inplace=True)

# Combine all symptom columns into one list of features
X = pd.get_dummies(df.drop("Disease", axis=1))

# Encode disease labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model trained successfully!")

