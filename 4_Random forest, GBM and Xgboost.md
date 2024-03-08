# Random Forest
- Random forest technique is one of the ensemble techniques where we use multiple decision trees for predicting
- Many decision trees are trained on different portions of data.
- When you need to predict, the input falls into each of the decision trees and the final prediction is based on majority voting or mean of all the decisions.

## Steps to create a random forest:
Loading the dataset and importing libraries
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load sample data
iris = load_iris()
X = iris.data  # Feature data
y = iris.target  # Target labels
```
```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a random forest classifier 
rf_model = RandomForestClassifier(n_estimators=20) # Number of decision trees in random forest  
```
```python
# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)
```

```python
# Evaluate (a simple example)
accuracy = rf_model.score(X_test, y_test) 
print("Accuracy:", accuracy) 
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/9b73b884-e0dd-4bf1-80bf-b14e6e9941f1)

# XgBoost:
You're absolutely right! I apologize for the misunderstanding. Here's a revised version that adheres more closely to standard Markdown formatting:

**XGBoost: The Superstar of Gradient Boosting**

**XGBoost**, short for "Extreme Gradient Boosting", is a powerful and popular machine learning algorithm. It utilizes decision tree-based ensemble methods.

**1. Building on Success**

* **Gradient Boosting Foundation:** XGBoost builds upon gradient boosting, where an ensemble of weak learners (usually decision trees) correct the errors of previous models. 
* **XGBoost's Enhancements:**
    * **Regularization:** Prevents overfitting.
    * **Efficient Handling of Missing Values:** Built-in strategy for data gaps.
    * **Speed and Parallel Tree Building:** Optimized for speed and large datasets.
    * **Flexibility:** Supports classification, regression, and ranking problems.

**2. Why is XGBoost so Popular?**

* **Performance:**  Known for its accuracy in real-world machine learning applications.
* **Scalability:** Handles large datasets, can be distributed across machines.
* **Versatility:**
    * **Classification:** Predicts categories (e.g., email spam vs. not spam)
    * **Regression:** Predicts continuous values (e.g., stock prices)
    * **Ranking:** Orders items (e.g., search results)

**3. How it Works (Simplified)**

1. **First Tree:** Begins with a basic decision tree for initial predictions.
2. **Calculating Errors:** Identifies the errors (residuals) of the initial tree.
3. **Next Tree... Fix the Errors!** A new tree attempts to predict the residuals from the previous tree.
4. **Combine and Improve:** Predictions are combined, each tree improves the model.

**4. In the Real World**

* **Finance:** Credit risk prediction, stock prices
* **E-commerce:** Product recommendations, fraud detection 
* **Healthcare:** Disease prediction and diagnosis 
* **Advertising:** Click-through rate prediction 

**Let me know if you have any other formatting requirements or if you'd like examples of how this Markdown code would appear when rendered!** 

#### implementation
Absolutely! Here's a simple implementation of XGBoost using Python and the scikit-learn library. We'll focus on a classification task, but the process is very similar for regression.

**1. Setup**

Make sure you have these libraries installed:

```bash
pip install xgboost scikit-learn pandas numpy
```

**2. Load a Dataset**

We'll use the classic Iris dataset for demonstration. You can load it directly from scikit-learn:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data  # Features 
y = iris.target  # Target labels
```

**3. Split Data**

Split your dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
```

**4. Initialize the XGBoost Model**

```python
from xgboost import XGBClassifier

model = XGBClassifier() 
```

**5. Train the Model**

```python
model.fit(X_train, y_train)
```

**6. Make Predictions**

```python
y_pred = model.predict(X_test) 
```

**7. Evaluate Performance**

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/6cc3cb35-b4c7-4c3d-947e-33df6b54401e)

# **GBM: The Ensemble Maestro**

Gradient Boosting Machines belong to a powerful class of machine learning algorithms that combine multiple weak learners (usually decision trees) into a stronger overall model. Here's the core idea:

1. **A Series of Weak Learners:** Instead of building one complex decision tree to try to capture everything at once, GBM starts with a very simple, even inaccurate decision tree. This initial tree makes some predictions.

2. **Error Analysis:** The errors (residuals) of these initial predictions are carefully examined.

3. **New Tree: Focusing on Errors:** The next decision tree is created to focus specifically on predicting those errors. Essentially, this new tree is trying to learn from the mistakes of the first.

4. **Combine and Improve:** The predictions of the first tree and the "error-correcting" tree are combined. We repeat this process, gradually adding more trees, each focusing on the remaining errors.

**Why Gradient Boosting?**

* **Iterative Refinement:** The sequential, error-correcting approach leads to a final model that's better than a single, large tree. Think of it as teamwork!
* **Gradients:** The name comes from how it uses "gradients" during optimization to figure out how to best focus each new tree.
* **Flexibility:** GBM can handle classification (predicting categories) and regression (predicting continuous values) tasks.

**Advantages of GBM**

* **Accuracy:** GBMs often achieve excellent performance on many real-world datasets.
* **Handles Missing Data:**  GBMs have strategies for dealing with gaps in your data.
* **Handles Non-linearity:** The tree-based structure can capture complex patterns that linear models might miss.

**Key Considerations**

* **Overfitting:** GBMs can be prone to overfitting if not careful. Hyperparameter tuning (controlling things like tree depth, the number of trees) and regularization help prevent this.
* **Computational Cost:**  Training many trees can become computationally expensive, especially for large datasets.

**Where GBMs Shine**

GBMs excel in many applications. Examples include:

* **Structured/Tabular Data:**  They do exceptionally well on typical datasets represented by rows and columns.
* **Predicting Customer Behavior:**  Modeling customer churn, product preferences, etc.
* **Risk Analysis:**  Fraud detection, credit scoring in finance.

Absolutely! Here's a simple implementation of a Gradient Boosting Machine (GBM) using Python and the scikit-learn library.

**1. Setup**

Ensure you have the scikit-learn library installed:

```bash
pip install scikit-learn 
```

**2. Load Sample Dataset**

We'll use the classic Iris dataset for demonstration purposes:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
```

**3. Split Data**

Split your dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**4. Initialize the GBM Model**

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier() 
```

**5. Fit the Model (Training)**

```python
model.fit(X_train, y_train)
```

**6. Make Predictions**

```python
y_pred = model.predict(X_test)
```

**7. Evaluate**

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

**Complete Code Example:**

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = GradientBoostingClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/1b657ee1-e2bf-4fee-a0ac-ab661c07c3e2)

