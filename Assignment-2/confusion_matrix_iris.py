import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate with Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("🧮 Confusion Matrix:\n", cm)

# 6. Calculate Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n📊 Evaluation Metrics:")
print(f"✅ Accuracy:  {acc:.2f}")
print(f"✅ Precision: {prec:.2f}")
print(f"✅ Recall:    {rec:.2f}")
print(f"✅ F1-score:  {f1:.2f}")

# 7. Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Iris Classification")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=300)
plt.show()
