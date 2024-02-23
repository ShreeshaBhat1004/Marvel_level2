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
