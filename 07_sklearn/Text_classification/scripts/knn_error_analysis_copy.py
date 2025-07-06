#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check current working directory (for debugging relative paths)
print("Current working directory:", os.getcwd())

# === Load Data ===
# Modify this path to the correct absolute or relative path of your dataset
data_path = '../../../12_data/data.csv'

try:
    df = pd.read_csv(data_path, index_col=0)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at path: {data_path}")

# === Feature Scaling ===
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit scaler on features (exclude target) and transform
features = df.drop('TARGET CLASS', axis=1)
scaled_features = scaler.fit_transform(features)

# Create scaled features DataFrame
df_feat = pd.DataFrame(scaled_features, columns=features.columns)

# === Train-Test Split ===
from sklearn.model_selection import train_test_split

X = df_feat
y = df['TARGET CLASS']

# Split data: 70% train, 30% test, fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# === Train KNN Classifier (k=1) ===
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print("Confusion Matrix for k=1:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report for k=1:")
print(classification_report(y_test, pred))

# Visualize confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (k=1)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# === Find Optimal K by Error Rate ===
error_rate = []

for k in range(1, 40):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    pred_k = knn_k.predict(X_test)
    error_rate.append(np.mean(pred_k != y_test))

# Plot Error Rate vs. K Value
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# === Retrain with Best K ===
best_k = error_rate.index(min(error_rate)) + 1  # index 0-based
print(f"Best K found: {best_k}")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
pred_best = knn_best.predict(X_test)

print(f"\nConfusion Matrix for k={best_k}:")
print(confusion_matrix(y_test, pred_best))
print(f"\nClassification Report for k={best_k}:")
print(classification_report(y_test, pred_best))

# Visualize confusion matrix for best k
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, pred_best), annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (k={best_k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
