# ============================================
# Experiment-11: Ensemble Voting Classifier
# ============================================
# This experiment demonstrates ensemble learning using a Voting Classifier
# It combines Logistic Regression, SVM, and Random Forest classifiers

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset for demonstration
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual classifiers
logistic_clf = LogisticRegression(random_state=42)
svm_clf = SVC(probability=True, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create an ensemble using a VotingClassifier
ensemble_clf = VotingClassifier(estimators=[('lr', logistic_clf), ('svm', svm_clf), ('rf', rf_clf)], voting='soft')

# Train the ensemble classifier
ensemble_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble_clf.predict(X_test)

# Evaluate the ensemble classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Accuracy: {accuracy:.2f}')