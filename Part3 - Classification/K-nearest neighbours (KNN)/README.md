# K-Nearest Neighbors (KNN)

## 📌 Introduction
K-Nearest Neighbors (KNN) is a **supervised machine learning algorithm** used for both **classification** and **regression** tasks. It predicts results based on the similarity (distance) between data points.

KNN is known as a **lazy learning algorithm** because it does not build a model during training; instead, it stores the dataset and performs computation during prediction.

---

## 🧠 How KNN Works
1. Choose the value of **K** (number of neighbors)
2. Calculate the distance between the test point and all training points
3. Select the **K nearest neighbors**
4. Make prediction:
   - **Classification** → Majority voting
   - **Regression** → Average of neighbors

---

## 📐 Distance Metrics
- **Euclidean Distance**
- **Manhattan Distance**
- **Minkowski Distance**
- **Cosine Similarity**

---

## Feature Scaling
Feature scaling is important in KNN because it relies on distance calculations.

Common techniques:
Min-Max Scaling
Standardization (Z-score)

## Disadvantages
1. Large datasets
2. Sensitive to outliers
3. Non-homogeneous scales
4. Imbalance dataset
5. Inference and not for prediction 
