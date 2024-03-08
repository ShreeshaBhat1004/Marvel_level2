## Ensemble techniques
Ensemble techniques are used to generate outputs with greater accuracy by using multiple models 
and taking out weighted mean, majority voting or training another model through obtained results. 

There are 3 Types of ensemble technqies these are:
1. Stacking
2. Bagging
3. Boosting

### Stacking
Stacking, an ensemble method, is often referred to as stacked generalization. This technique works by allowing a training algorithm to ensemble several other similar learning algorithm predictions. Stacking has been successfully implemented in regression, density estimations, distance learning, and classifications. It can also be used to measure the error rate involved during bagging.

#### Steps
*Train a bunch of models*: Start by training various different models on your dataset (these could be decision trees, support vector machines, neural networks... the more diverse, the better).
Make predictions on new data: Get each of your trained models to predict on a new set of data points (not the ones they trained on).
Train the super-learner: Now you create a new dataset where the features are the predictions from those first models, and the target is still what you're actually trying to predict. You train a final model (any kind you want!) on this special dataset.
Predict!: When faced with completely new data, you first get predictions from all your initial models, then feed those predictions into the super-learner model for the ultimate decision.
```
                              ┌─────────────────────────────────────────┐
                               │               Final Model               │
                               │            (Meta-level Model)           │
                               │   ┌────────────────────────────────────┐ │
                               │   │  Combine Predictions from Base     │ │
                               │   │  Models to make Final Prediction   │ │
                               │   └────────────────────────────────────┘ │
                               └───────────────────┬─────────────────────┘
                                                   │
                                                   │
       ┌───────────────────────────────────────────┴────────────────────────────────────────┐
       │                                           │                                        │
       │                                           │                                        │
┌──────┴──────┐                            ┌──────┴──────┐                          ┌──────┴──────┐
│ Base Model 1│                            │ Base Model 2│                          │ Base Model 3│
│  (e.g., SVM)│                            │  (e.g., RF) │                          │ (e.g., kNN) │
└─────────────┘                            └─────────────┘                          └─────────────┘
       │                                           │                                        │
       │                                           │                                        │
       │                                           │                                        │
       ▼                                           ▼                                        ▼
┌──────────────┐                          ┌──────────────┐                        ┌──────────────┐
│ Predictions 1│                          │ Predictions 2│                        │ Predictions 3│
└──────────────┘                          └──────────────┘                        └──────────────┘
       │                                           │                                        │
       │                                           │                                        │
       └───────────────────────────────────────────┴────────────────────────────────────────┘
                                                   │
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │    Stacked   │
                                          │  Predictions │
                                          └──────────────┘
```
#### Implementation
We import neccessary libraries
```python
import pandas as pd 
from sklearn.datasets import load_iris # Our dataset
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression # ML model 1
from sklearn.neighbors import KNeighborsClassifier # ML model 2
from sklearn.svm import SVC # ML Model 3
from sklearn.ensemble import StackingClassifier # Method that helps in stacking

# Load a sample dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
We create a list of tuples that represents our base models and also it is onwhich we are going to use for stacking
```python
# Pick some diverse models
base_models = [
    ('KNN', KNeighborsClassifier(n_neighbors=3)),
    ('SVM', SVC(kernel='linear')), 
    ('Logistic', LogisticRegression(max_iter=200)) 
]
```
We train a meta learner or a super learner, where its training dataset has one more coloumn and that is predictions made from base models.
```python
# Use logistic regression as our meta-learner (you can experiment with others)
meta_model = LogisticRegression(max_iter=200) 

# The stacking classifier itself
ensemble = StackingClassifier(estimators=base_models, final_estimator=meta_model)
```
Finally we test the model
```python
# Fit the ensemble on the training data
ensemble.fit(X_train, y_train)

# Use the ensemble for predictions on new data
y_pred = ensemble.predict(X_test)
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/ebab91ca-c99c-4f62-874c-c101d4a8b9a6)

### Bagging: 
Bagging is an ensemble learning technique designed to reduce the variance of a machine learning model, making it less prone to overfitting. Here's how it works:

#### Steps:
Bootstrapping: Create multiple different subsets of your original training data by sampling with replacement. This means some data points might be included in a subset multiple times, while others won't appear at all.
Train individual models: Train a base model (e.g., a decision tree) on each of these bootstrapped subsets. This forces each model to specialize on different aspects of the data distribution.
Aggregation: When making predictions on new data, take the predictions of each individual model and combine them. This is typically done using a majority vote (for classification) or by averaging (for regression).
```
                               ┌───────────────────────────────────────────────┐
                               │                Final Prediction               │
                               │      (Aggregation of Base Model Predictions)  │
                               └───────────────────────┬───────────────────────┘
                                                       │
                                                       │
       ┌───────────────────────────────────────────────┴───────────────────────────────────────────┐
       │                                               │                                           │
       │                                               │                                           │
┌──────┴──────┐                                 ┌──────┴──────┐                             ┌──────┴──────┐
│ Base Model 1│                                 │ Base Model 2│                             │ Base Model m│
└─────────────┘                                 └─────────────┘                             └─────────────┘
       │                                               │                                           │
       │                                               │                                           │
       │                                               │                                           │
       ▼                                               ▼                                           ▼
┌──────────────┐                               ┌──────────────┐                             ┌──────────────┐
│  Bootstrap   │                               │  Bootstrap   │                             │  Bootstrap   │
│  Sample 1    │                               │  Sample 2    │                             │  Sample m    │
└──────────────┘                               └──────────────┘                             └──────────────┘
       │                                               │                                           │
       │                                               │                                           │
       │                                               │                                           │
       ▼                                               ▼                                           ▼
┌──────────────┐                               ┌──────────────┐                             ┌──────────────┐
│ Prediction 1 │                               │ Prediction 2 │                             │ Prediction m │
└──────────────┘                               └──────────────┘                             └──────────────┘
       │                                               │                                           │
       │                                               │                                           │
       └───────────────────────────────────────────────┴───────────────────────────────────────────┘
                                                       │
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │   Average/   │
                                                │   Majority   │
                                                │    Vote      │
                                                └──────────────┘
```
#### Implementation
We first import all neccessary libraries, load the dataset and split it as X and y that means, all the feature columns as x and our target as y
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer # Our dataset
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import BaggingClassifier # Model 1
from sklearn.tree import DecisionTreeClassifier # Model 2
from sklearn.metrics import accuracy_score 

# Sample dataset: Breast cancer classification
data = load_breast_cancer()
X = data.data
y = data.target

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10) 
```
If we just use a decision tree, what's the accuracy
```python
# A single decision tree as our comparison point
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Accuracy of single decision tree:", accuracy_score(y_test, y_pred))
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/2cdad622-6aae-4c4e-882a-18f156952276)

Now we use bagging ensemble technique and calculate the efficiency
```python
# Base estimator still a decision tree (but with high flexibility for bagging's sake)
base_estimator = DecisionTreeClassifier(max_depth=None)

# Create the ensemble
ensemble = BaggingClassifier(base_estimator=base_estimator, n_estimators=100, random_state=10)

# Fit the ensemble model 
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)

print("Accuracy of Bagging ensemble: ", accuracy_score(y_test, y_pred_ensemble))
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/19ff0955-3874-4c3b-b24a-07ae22ccfddd)

### Boosting
Boosting is an ensemble technique that seeks to reduce bias and improve the accuracy of weak learners by creating a sequence of models where each one learns from the mistakes of the previous one.

### Steps

Start with a weak learner: A weak learner is a model that performs slightly better than random guessing. Often, decision trees with limited depth are used.
Train and focus on mistakes: Train the first model on the dataset. The data points that were misclassified get more weight (become more important) for the next model.
Train the next model: Now, a new model is trained while paying extra attention to the examples the previous model got wrong. This forces the new model to focus on those hard-to-classify cases.
Repeat and combine: This process repeats, each time creating a new model that tries to fix the shortcomings of the ensemble so far. Finally, the predictions from all models are combined, usually with a weighted scheme where better-performing models get more say in the final outcome.
```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
       │                                            Final Model                                            │
       │                              (Weighted Combination of Weak Learners)                              │
       └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                                          ┌────────────────────────────┐
                                          │       Weak Learner 1       │
                                          │      (e.g., Decision Stump)│
                                          └────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                                          ┌────────────────────────────┐
                                          │       Weak Learner 2       │
                                          │      (e.g., Decision Stump)│
                                          └────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                                                       ...
                                                        │
                                                        │
                                                        ▼
                                          ┌────────────────────────────┐
                                          │       Weak Learner m       │
                                          │      (e.g., Decision Stump)│
                                          └────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                                               ┌───────────────┐
                                               │   Weighted    │
                                               │  Combination  │
                                               └───────────────┘
  ```

#### Implementation
Loading the dataset
```python
import numpy as np
from sklearn.datasets import load_iris 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```
```python
# Base weak learner (Decision Tree)
dt_base = DecisionTreeClassifier(max_depth=1)  # Simple tree, weak learner

# AdaBoost ensemble
boost = AdaBoostClassifier(base_estimator=dt_base, n_estimators=50) 
boost.fit(X_train, y_train)

# Predictions and accuracy
y_pred_boost = boost.predict(X_test)
accuracy_boost = accuracy_score(y_test, y_pred_boost)

# Single decision tree for comparison
dt_single = DecisionTreeClassifier(max_depth=1)
dt_single.fit(X_train, y_train)
y_pred_single = dt_single.predict(X_test)
accuracy_single = accuracy_score(y_test, y_pred_single)

print("Accuracy of single decision tree:", accuracy_single)
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/25440d10-2903-4ebb-a8ee-9f18849796db)

```python
print("Accuracy of boosted ensemble:", accuracy_boost)
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/8fb9ca9e-3537-487f-9e7f-04020849faf7)

