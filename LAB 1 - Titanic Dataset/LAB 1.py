import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the Titanic dataset from a local file
# noinspection PyArgumentList
titanic = pd.read_csv("Titanic-Dataset.csv")

# Data Preprocessing
# Drop unnecessary columns
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values using explicit assignment
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])

# Convert categorical columns to numerical
label_encoders = {}
for column in ['Sex', 'Embarked']:
    le = LabelEncoder()
    titanic[column] = le.fit_transform(titanic[column])
    label_encoders[column] = le

# Define feature variables (X) and target variable (y)
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables (for algorithms like SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Support Vector Classifier": SVC(kernel='linear', random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Use scaled features for SVC, otherwise use unscaled features
    if name == "Support Vector Classifier":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived'], output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    # Store the results
    results[name] = {"Accuracy": accuracy, "Classification Report": report, "Confusion Matrix": confusion}

# Create a DataFrame for the summary accuracy results
summary_df = pd.DataFrame({name: {"Accuracy": metrics["Accuracy"]} for name, metrics in results.items()}).T
print("Titanic Dataset Model Accuracy Comparison")
print(summary_df)

# Visualization of results
model_names = list(results.keys())
accuracies = [metrics["Accuracy"] for metrics in results.values()]
f1_scores = [metrics["Classification Report"]["weighted avg"]["f1-score"] for metrics in results.values()]

# Set up bar width and positions
bar_width = 0.35
index = np.arange(len(model_names))

# Plot accuracy scores
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = plt.bar(index, accuracies, bar_width, label='Accuracy', color='skyblue')

# Plot F1-scores
bar2 = plt.bar(index + bar_width, f1_scores, bar_width, label='F1-Score', color='salmon')

# Add labels, title, and legend
plt.xlabel('Machine Learning Models')
plt.ylabel('Scores')
plt.title('Model Performance on Titanic Dataset: Accuracy vs F1-Score')
plt.xticks(index + bar_width / 2, model_names, rotation=45)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
