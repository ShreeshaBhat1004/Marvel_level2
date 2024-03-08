## **Anomaly Detection** 
Anomaly detection is the technique of identifying rare events or observations which can raise suspicions by being statistically different from the rest of the observations. Such “anomalous” behaviour typically translates to some kind of a problem like a credit card fraud, failing machine in a server, a cyber attack, etc.

An anomaly can be broadly categorized into three categories – \n
**Point Anomaly:** A tuple in a dataset is said to be a Point Anomaly if it is far off from the rest of the data.\n
**Contextual Anomaly**: An observation is a Contextual Anomaly if it is an anomaly because of the context of the observation.\n
**Collective Anomaly:** A set of data instances help in finding an anomaly.

Anomaly detection can be done using the concepts of Machine Learning. It can be done in the following ways –\n

##### **Supervised Anomaly Detection:**
This method requires a labeled dataset containing both normal and anomalous samples to construct a predictive model to classify future data points. The most commonly used algorithms for this purpose are supervised Neural Networks, Support Vector Machine learning, K-Nearest Neighbors Classifier, etc.
#####**Unsupervised Anomaly Detection:**
This method does require any training data and instead assumes two things about the data ie Only a small percentage of data is anomalous and Any anomaly is statistically different from the normal samples. Based on the above assumptions, the data is then clustered using a similarity measure and the data points which are far off from the cluster are considered to be anomalies.

#### Implementation:
```python
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load your dataset (replace 'your_data.csv')
data = pd.read_csv('/content/sample_data/Anomaly Detection_1/annthyroid_21feat_normalised.csv')

# Create an Isolation Forest model
model = IsolationForest(contamination=0.05)  # Assume 5% of data points are anomalies

# Fit the model to your data
model.fit(data)

# Get anomaly scores (-1 is normal, 1 is an outlier)
anomaly_scores = model.decision_function(data) 

# Identify anomalies (adjust threshold as needed)
anomalies = data[anomaly_scores > 0.17] # We declare that datapoints that are above 1.7 are anomalies
print(anomalies)
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/a33951b0-1909-49b9-8095-590b7c5c56c4)


