# Machine Learning Data Preprocessing

This project demonstrates the **data preprocessing pipeline** for machine learning in Python, including handling missing data, encoding categorical variables, splitting the dataset, and feature scaling.

---

## 1️⃣ Importing Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
**Explanation:**
- `numpy` (`np`) → For numerical operations and arrays.
- `matplotlib.pyplot` (`plt`) → For plotting and visualization.
- `pandas` (`pd`) → For data manipulation and DataFrames.

---

## 2️⃣ Loading the Dataset
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```
**Explanation:**
- Loads CSV data into a DataFrame.
- `X` → all columns except last (features).
- `y` → last column (target).

---

## 3️⃣ Handling Missing Data
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```
**Explanation:**
- Fills missing values (`NaN`) with column mean.
- Applied to numeric feature columns (here columns 1 and 2).

---

## 4️⃣ Encoding Categorical Data
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```
**Explanation:**
- One-Hot Encoding converts categorical columns (like 'Country') into binary columns.
- Other columns remain unchanged.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```
- Label encodes target variable `y` (e.g., 'Yes' → 1, 'No' → 0).

---

## 5️⃣ Splitting Dataset into Training and Test Sets
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```
**Explanation:**
- Splits data into training (80%) and test (20%) sets.
- `random_state=1` ensures reproducibility.

---

## 6️⃣ Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```
**Explanation:**
- Standardizes numeric features to have mean=0 and std=1.
- `fit_transform()` → computes mean/std on training set and scales it.
- `transform()` → scales test set using **training statistics**.
- Scaling is applied **only to features**, not the target.

---

## ✅ Summary of Preprocessing Steps
1. Import libraries → NumPy, Pandas, Matplotlib
2. Load dataset → Read CSV into DataFrame
3. Split features and target → X and y
4. Handle missing data → Fill NaN with mean
5. Encode categorical data → One-Hot for X, LabelEncoder for y
6. Split dataset → Training and test sets
7. Feature scaling → Standardize numeric features

This pipeline prepares your dataset to be **ready for training machine learning models**.