# ============================================================
# Task 1 - Iris Flower Classification
# Internship: CodTech IT Solutions Pvt. Ltd.
# Domain: Artificial Intelligence
# ============================================================

# ---------- Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

# ---------- 1. Load Dataset ----------
print("=" * 55)
print("       IRIS FLOWER CLASSIFICATION")
print("=" * 55)

df = pd.read_csv('iris.csv')

print("\n[1] Dataset Overview")
print(f"    Shape : {df.shape}")
print(f"    Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# ---------- 2. Exploratory Data Analysis ----------
print("\n[2] Basic Statistics")
print(df.describe())

print("\n[3] Class Distribution")
print(df['species'].value_counts())

print("\n[4] Missing Values")
print(df.isnull().sum())

# ---------- 3. Visualizations ----------

# 3a. Pairplot
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', palette='Set1')
plt.suptitle('Iris - Pairplot of Features', y=1.02, fontsize=14)
plt.savefig('pairplot.png', bbox_inches='tight')
plt.close()
print("\n[5] Pairplot saved as 'pairplot.png'")

# 3b. Correlation Heatmap
plt.figure(figsize=(7, 5))
numeric_df = df.drop(columns=['species'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()
print("[6] Heatmap saved as 'correlation_heatmap.png'")

# 3c. Boxplot for each feature
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_df.columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=col, data=df, palette='pastel')
    plt.title(f'{col} by Species')
plt.tight_layout()
plt.savefig('boxplots.png')
plt.close()
print("[7] Boxplots saved as 'boxplots.png'")

# ---------- 4. Preprocessing ----------
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[8] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ---------- 5. Model Training & Evaluation ----------
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42)
}

results = {}
print("\n[9] Model Accuracies")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  {name:<30} : {acc * 100:.2f}%")

# ---------- 6. Best Model Details ----------
best_model_name = max(results, key=results.get)
best_model      = models[best_model_name]
y_pred_best     = best_model.predict(X_test)

print(f"\n[10] Best Model: {best_model_name} ({results[best_model_name]*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# ---------- 7. Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("\n[11] Confusion matrix saved as 'confusion_matrix.png'")

# ---------- 8. Model Comparison Bar Chart ----------
plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), [v * 100 for v in results.values()],
               color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
plt.ylim(80, 105)
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison - Iris Classification')
plt.xticks(rotation=20, ha='right')
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("[12] Model comparison chart saved as 'model_comparison.png'")

# ---------- 9. Sample Prediction ----------
print("\n[13] Sample Prediction on New Data")
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                      columns=['sepal_length', 'sepal_width',
                                'petal_length', 'petal_width'])
pred_encoded = best_model.predict(sample)
pred_class   = le.inverse_transform(pred_encoded)
print(f"    Input  : {sample.values.tolist()[0]}")
print(f"    Predicted Species : {pred_class[0]}")

print("\n" + "=" * 55)
print("  Task Completed Successfully!")
print("=" * 55)
