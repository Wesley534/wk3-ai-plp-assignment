import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

# --- 1. Load and Preprocess the Data ---
try:
    # Use the file name from your environment
    df = pd.read_csv('Iris.csv')
except FileNotFoundError:
    print("Error: 'Iris.csv' not found. Please upload it to your Replit project.")
    exit()

# Drop 'Id' and correctly capitalize 'Species'
X = df.drop(['Id', 'Species'], axis=1)
y_species = df['Species']

# Encode Labels (Target variable: Species)
le = LabelEncoder()
y = le.fit_transform(y_species)

# Save the LabelEncoder and Target Names (needed for prediction output)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)
target_names = le.classes_

# Split (not strictly needed for final model, but good practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Train the Model ---
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# --- 3. Save the Model ---
# Serialize the trained model to a file
model_filename = 'model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(dt_classifier, file)

print(f"Model trained and saved as {model_filename}")
print(f"Label Encoder saved as label_encoder.pkl")