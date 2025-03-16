# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
file_path = "C:\\Users\\Yogesh V\\Downloads\\home_loan_data.csv"
df = pd.read_csv(file_path)

# Step 2: Handle missing values (fill missing numerical values with median)
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 3: Define features (X) and target variable (y)
X = df.drop(columns=['LoanStatus'])  # Features
y = df['LoanStatus']  # Target variable (Loan Approved/Rejected)

# Step 4: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Normalize/Scale the features to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)  # Transform test data using the same scaler

# Step 6: Train an Optimized Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=300,        # Increased trees for better learning
    max_depth=15,            # Control tree depth to prevent overfitting
    min_samples_split=5,     # Reduce overfitting by requiring minimum samples
    class_weight="balanced", # Handle imbalanced data (if any)
    random_state=42
)
model.fit(X_train, y_train)  # Train the model

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate model performance using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.2f}")  # Display accuracy percentage

# Step 9: Save the trained model and scaler for deployment
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Optimized Model and Scaler saved successfully!")
