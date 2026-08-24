# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load the MNIST dataset with cache disabled to avoid checksum issues
mnist = fetch_openml('mnist_784', version=1, cache=False)
X = mnist.data
y = mnist.target.astype(int)  # Convert target to integer

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Support Vector Classifier": SVC(kernel='linear', random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Use scaled features for all models due to high-dimensional data
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    # Store the results
    results[name] = {"Accuracy": accuracy, "Classification Report": report, "Confusion Matrix": confusion}

# Create a DataFrame for the summary accuracy results
summary_df = pd.DataFrame({name: {"Accuracy": metrics["Accuracy"]} for name, metrics in results.items()}).T

# Display the summary accuracy DataFrame
print("MNIST Dataset Model Accuracy Comparison")
print(summary_df)

# Output full classification reports and confusion matrices
for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("\nClassification Report:")
    for label, score in metrics["Classification Report"].items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            print(f"  Class: {label}")
            for metric, value in score.items():
                print(f"    {metric.capitalize()}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["Confusion Matrix"])

# Code to generate the bar plot for accuracy and F1-scores
model_names2 = list(results.keys())
accuracies = [metrics["Accuracy"] for metrics in results.values()]
f1_scores = [metrics["Classification Report"]["weighted avg"]["f1-score"] for metrics in results.values()]

# Set up bar width and positions
bar_width = 0.35
index = np.arange(len(model_names2))

# Plot accuracy scores
fig, ax = plt.subplots(figsize=(12, 6))
bar1 = plt.bar(index, accuracies, bar_width, label='Accuracy', color='lightgreen')

# Plot F1-scores
bar2 = plt.bar(index + bar_width, f1_scores, bar_width, label='F1-Score', color='orange')

# Add labels, title, and legend
plt.xlabel('Machine Learning Models')
plt.ylabel('Scores')
plt.title('Model Performance on MNIST Dataset: Accuracy vs F1-Score')
plt.xticks(index + bar_width / 2, model_names2, rotation=45)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
