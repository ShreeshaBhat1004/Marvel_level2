# Decision tree based ID3 Algorithm
**Decision tree** algorithm is a branch of Machine learning, it is a recursive algorithm which traverses a dataset by keeping a root node and moving onto its branches or child nodes by only considering non-traversed nodes.We can perform regression(numerical prediction) tasks as well as Classification(yes or no prediction) tasks using Decision trees. 

**Why the algorithm is called *decision tree*?**
An analogy between a tree and a decision tree can help illustrate the concept of a decision tree. In a decision tree, you have a root node (analogous to the tree's trunk) that represents the initial decision point. From the root node, there are branches (analogous to the tree's branches) that lead to different decisions or choices. These branches are split based on specific conditions or features related to the data. As you move down the tree, you encounter more nodes and branches, much like the tree's branches and leaves. Ultimately, you reach the leaf nodes (analogous to the tree's leaves), which represent the final outcomes or decisions.
________________________________________________________________________________________________________
### Basic terminologies of Decision trees: 
Decision trees are a popular machine learning algorithm used for both classification and regression tasks. Understanding some basic terminologies associated with decision trees is essential. Here are the key terms:

1. **Root Node:** The topmost node in a decision tree, from which the tree starts branching. It represents the entire dataset.

2. **Internal Node (or Split Node):** A node in the decision tree that has child nodes, meaning it represents a decision point where the data is split based on a particular feature.

3. **Leaf Node (or Terminal Node):** A node in the decision tree that has no child nodes, and it represents the final prediction or decision. In classification, a leaf node corresponds to a class label, and in regression, it contains a numerical value.

4. **Splitting:** The process of dividing the data at an internal node into two or more child nodes based on a feature's value. The feature and its corresponding value that determine the split are called a "splitting criterion."

5. **Splitting Criterion (or Splitting Rule):** The feature and value used to split the data at an internal node. The most common splitting criteria include Gini impurity, entropy, and mean squared error.

6. **Entropy:** A measure of impurity in a node. It quantifies the likelihood of misclassifying a randomly chosen element in the dataset if it were classified according to the class distribution in that node. It is used in classification problems.

7. **Information Gain:** The reduction in impurity achieved by splitting a node using a particular splitting criterion. Decision trees aim to maximize information gain when choosing the best feature for splitting.

8. **Pruning:** The process of reducing the size of a decision tree by removing branches (subtrees) that do not provide significant improvements in prediction. Pruning helps prevent overfitting.

9. **Maximum Depth:** The maximum number of levels or nodes from the root node to a leaf node in the decision tree. It is a hyperparameter that can be used to control the tree's complexity.

10. **Feature Importance:** A measure of the relevance or importance of each feature in the decision tree for making predictions. It helps identify which features have the most impact on the model's decisions.

These are some of the fundamental terminologies associated with decision trees. Understanding these terms is essential when working with decision tree algorithms and interpreting the results of decision tree models.
# One diagram that represents all the terms and also mathematical formula below the terms.
____________________________________________________________________________
### What's ID3(Iterative Dichotomiser 3)?
ID3 is a machine learning algorithm used for building decision trees from a given dataset. It employs a top-down, recursive approach to partition the dataset into subsets, selecting the most informative attributes to split the data at each step. The goal of ID3 is to create a decision tree that can be used for classification by dividing the data into subsets that are as pure as possible in terms of the target class. The algorithm is based on information theory and uses entropy as a measure to select the best attributes for splitting the data.
_____________________________________________________________________________________________________________________________________________________________
### Implementation of ID3 algorithm 
**Step 1: Observing the dataset**
We will look into a very famous dataset for ID3 called "Play tennis" dataset. 
| Outlook | Temperature | Humidity | Wind | Play Tennis |
|---|---|---|---|---|
| Sunny | Hot | High | Weak | Yes |
| Sunny | Hot | High | Strong | No |
| Overcast | Hot | High | Weak | Yes |
| Rain | Mild | High | Weak | Yes |
| Rain | Cool | Normal | Weak | No |
| Rain | Cool | Normal | Strong | Yes |
| Overcast | Cool | Normal | Strong | Yes |
| Sunny | Mild | High | Weak | Yes |
| Sunny | Cool | Normal | Weak | No |
| Rain | Mild | Normal | Weak | Yes |
| Sunny | Mild | Overcast | Mild | High | Yes |
| Overcast | Hot | Normal | Yes | No |
| Rain | Mild | High | Strong | No |

After Observing the dataset we can see that the **features** are - Outlook,Temperature,Humidity,wind.
**label:** Play tennis

**Step 2: Importing the neccessary libraries**
We are going to use pandas library to manipulating the dataset and numpy library for mathematical calculation. 
``` python
import pandas as pd 
import numpy as np
```
**Step 3: Reading the dataset using pandas**
We are going to read the 'play tennis' dataset which is a csv file using pandas and convert it into a pandas dataframe using read(). 
```python
train_data = pd.read_csv("https://github.com/Hackinfinity/VTU-Machine-Learning-Lab-program-ID3-Algorithm/blob/master/PlayTennis.csv")
```
**Step 4: Caclulating the entropy of the whole dataset**
Entropy of a given dataset can be written as 
.
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/c5f634d9-5086-46ba-b9c0-176dc7ab01cd)

Now, Entropy of the whole dataset is\
Total row = 14\
Row with "Yes" class = 9\
Row with "No" class = 5\
Complete entropy of dataset is -
H(S) = - p(Yes) * log2(p(Yes)) - p(No) * log2(p(No))\
     = - (9/14) * log2(9/14) - (5/14) * log2(5/14)\
     = - (-0.41) - (-0.53)\
     = 0.94

```python
def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    total_entr = 0
    
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #entropy of the class
        total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset
    
    return total_entr
```
**Step 5: Calculating the entropy for each feature or filtered dataset**
Categorical values of Outlook - Sunny, Overcast and Rain\
Total count of row containing:\
  Sunny = 5\
  Sunny & Yes = 2\
  Sunny & No = 3\
H(Outlook=Sunny) = -(2/5)*log(2/5)-(3/5)*log(3/5) = 0.971\
Total count of row containing:\  
  Rain = 5\
  Rain & Yes = 3\
  Rain & No = 2\
H(Outlook=Rain) = -(3/5)*log(3/5)-(2/5)*log(2/5) = 0.971\
Total count of row containing:\
  Overcast = 4\
  Overcast & Yes = 4\
  Overcast & No = 0\
H(Outlook=Overcast) = -(4/4)*log(4/4)-0 = 0

Note that we need to perform same to other features too, so we can just implement python code here to perform calculations for each feature

```python
def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy
```
**Step 6: Calculating information gain for a feature**
We can calulate information gain for a feature by this formula 
Information gain(feature) = Entropy of whole dataset -{p(condition1)* H(condition1) + p(condition2) *H(condition2)...+p(condition_n)*H(condition_n)}

In python 
```python
def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        
    return calc_total_entropy(train_data, label, class_list) - feature_info #calculating information gain by subtracting
```
**Step 7: Finding the most informative feaature(feature with high information gain)**
Using the code above we can quickly calculate information gain for each feature. 
Information gain:\
  Outlook = 0.247 (Highest value)\
  Temperature = 0.0292\
  Humidity = 0.153\
  Wind = 0.048
These are the information gain of all the features. 

A python code to find the feature with highest information gain
```python

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
                                            #N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature
```
**Step 8: Adding a node to the tree**
As we have found the feature name with the highest information gain(outlook), we have to generate a node in the tree and its value as a branch. For example, we have selected Outlook, so we have to add Outlook as a node in the tree and its value Sunny or Rain or Overcast as a branch.

If any value of the feature represents only one class (Ex. only rows with Play Tennis = ‘Yes’ or ‘No’) then we can say that the feature value represents a pure class. If the value does not represent a pure value, we have to extend it further until we find a pure class.

Outlook is selected as Node.\
(Outlook = Sunny): Not pure class, contains both class Yes and No\
(Outlook = Overcast): Pure class, contains only one class Yes\
(Outlook = Rain): Not pure class, contains both class Yes and No

Now we have to remove the overcast value from the dataset so the dataset becomes

![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/0e97c034-958a-4ad9-9884-ac08d4d89c43)

Next we have to use the Updated dataset 
```python
def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node
    
    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value] #dataset with only feature_name = feature_value
        
        assigned_to_node = False #flag for tracking feature_value is pure class or not
        for c in class_list: #for each class
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] #count of class c

            if class_count == count: #count of (feature_value = count) of class (pure class)
                tree[feature_value] = c #adding node to the tree
                train_data = train_data[train_data[feature_name] != feature_value] #removing rows with feature_value
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            tree[feature_value] = "?" #as feature_value is not a pure class, it should be expanded further, 
                                      #so the branch is marking with ?
            
    return tree, train_data
```
**Step 9: Performing ID3 algorithm and generating tree**
Now, we have to set up algorithm such a way that it will recursively and repeatedly perform the step 4 - 8\
- Finding the most informative feature
- Making a tree node with a feature name and feature values as branches
- If pure class, adding leaf node (= Class) to the tree node
- If impure class, adding an expandable node (= ‘?’) to the tree node
- Shrinking/Updating the dataset according to the pure class
- Adding the node with branches into a tree
- Expand the branch of the next impure class (= ‘?’) with an updated dataset
The recursion endpoint:
- The dataset becomes empty after updating
- There is no expandable branch (= all pure class)
```python

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree(next_root, node, feature_value_data, label, class_list) #recursive call with updated dataset
```
**Step 10: Starting the algorithm and predicting from the tree**
To start the algorithm: 
```python

def id3(train_data_m, label):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree(tree, None, train_data, label, class_list) #start calling recursion
    return tree

tree = id3(train_data,'Play Tennis')

```
To Predict 
```python

def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None
```
Evaluating on a test Dataset
```python
# Creating a evaluate function that returns accuracy 
def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.iloc[index]) #predict the row
        if result == test_data_m[label].iloc[index]: #predicted value and expected value is same or not
            correct_preditct += 1 #increase correct count
        else:
            wrong_preditct += 1 #increase incorrect count
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return accuracy


test_data_m = pd.read_csv("test\PlayTennis.csv") #importing test dataset into dataframe

accuracy = evaluate(tree, test_data_m, 'Play Tennis') #evaluating the test dataset

```



