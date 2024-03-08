#### **Anomaly Detection** 
Anomaly detection is the technique of identifying rare events or observations which can raise suspicions by being statistically different from the rest of the observations. Such “anomalous” behaviour typically translates to some kind of a problem like a credit card fraud, failing machine in a server, a cyber attack, etc.

An anomaly can be broadly categorized into three categories – \n
**Point Anomaly:** A tuple in a dataset is said to be a Point Anomaly if it is far off from the rest of the data.\n
**Contextual Anomaly**: An observation is a Contextual Anomaly if it is an anomaly because of the context of the observation.\n
**Collective Anomaly:** A set of data instances help in finding an anomaly.

Anomaly detection can be done using the concepts of Machine Learning. It can be done in the following ways –\n

**Supervised Anomaly Detection:** This method requires a labeled dataset containing both normal and anomalous samples to construct a predictive model to classify future data points. The most commonly used algorithms for this purpose are supervised Neural Networks, Support Vector Machine learning, K-Nearest Neighbors Classifier, etc.\n
**Unsupervised Anomaly Detection:** This method does require any training data and instead assumes two things about the data ie Only a small percentage of data is anomalous and Any anomaly is statistically different from the normal samples. Based on the above assumptions, the data is then clustered using a similarity measure and the data points which are far off from the cluster are considered to be anomalies.
Step 1: Importing the required libraries\n

```Python3
import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt 
import matplotlib.font_manager 
from pyod.models.knn import KNN  
from pyod.utils.data import generate_data, get_outliers_inliers
```
Step 2: Creating the synthetic data
 

```Python3
# generating a random dataset with two features 
X_train, y_train = generate_data(n_train = 300, train_only = True, 
                                                   n_features = 2) 
  
# Setting the percentage of outliers 
outlier_fraction = 0.1
  
# Storing the outliers and inliners in different numpy arrays 
X_outliers, X_inliers = get_outliers_inliers(X_train, y_train) 
n_inliers = len(X_inliers) 
n_outliers = len(X_outliers) 
  
# Separating the two features 
f1 = X_train[:, [0]].reshape(-1, 1) 
f2 = X_train[:, [1]].reshape(-1, 1)
```
Step 3: Training and evaluating the model
 

```Python3
# Training the classifier 
clf = KNN(contamination = outlier_fraction) 
clf.fit(X_train, y_train) 
  
# You can print this to see all the prediction scores 
scores_pred = clf.decision_function(X_train)*-1
  
y_pred = clf.predict(X_train) 
n_errors = (y_pred != y_train).sum() 
# Counting the number of errors 
  
print('The number of prediction errors are ' + str(n_errors)) 
```

