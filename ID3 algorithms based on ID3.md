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

6. **Gini Impurity:** A measure of impurity in a node. It quantifies the likelihood of misclassifying a randomly chosen element in the dataset if it were classified according to the class distribution in that node. It is used in classification problems.

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
### Example implementation of ID3 on a custom dataset: 
The first thing we need would be our dataset: 
| Day | Weather | Temperature | Humidity | Wind | Play Football? |
|---|---|---|---|---|---|
| Day 1 | Sunny | Hot | High | Weak | No |
| Day 2 | Sunny | Hot | High | Strong | No |
| Day 3 | Cloudy | Hot | High | Weak | Yes |
| Day 4 | Rain | Mild | High | Weak | Yes |
| Day 5 | Rain | Cool | Normal | Weak | Yes |
| Day 6 | Rain | Cool | Normal | Strong | No |
| Day 7 | Cloudy | Cool | Normal | Strong | Yes |
| Day 8 | Sunny | Mild | High | Weak | No |
| Day 9 | Sunny | Cool | Normal | Weak | Yes |
| Day 10 | Rain | Mild | Normal | Weak | Yes |
| Day 11 | Sunny | Mild | Normal | Strong | Yes |
| Day 12 | Sunny | Hot | Normal | Strong | Yes |
| Day 13 | Rain | Mild | High | Strong | No |
| Day 14 | Sunny | Mild | High | strong | Yes | 

We have the above data taken in 14 days, and based on various features of weather, if the person has played football or not. Using these features, we can predict weather the person will play football or not given the weather conditions of the future. We can predict it using decision trees. 

As we know, a decision tree has a root node, internal nodes and then the leaf nodes. Now, how can we pick a root node to begin with. To find out which feature is going to be root node, we have to 
- Calculate entropy and information gain for each feature and the feature with highest ig will be root node.
- The entropy of the whole dataset will be
I apologize for the oversight. If there are 9 "Yes" and 5 "No" instances in the dataset, let's recalculate the entropy:

1. Number of "Yes" = 9
2. Number of "No" = 5

Now, calculate the probabilities:

- \(p(Yes) = \frac{\text{Number of "Yes"}}{\text{Total number of samples}} = \frac{9}{14}\)
- \(p(No) = \frac{\text{Number of "No"}}{\text{Total number of samples}} = \frac{5}{14}\)

Now, calculate the entropy:

\[Entropy(S) ≈ - (0.6429 * log2(0.6429)) - (0.3571 * log2(0.3571))\]

Using base 2 logarithm:

\[Entropy(S) ≈ - (0.6429 * (-0.6825)) - (0.3571 * (-1.5144))\]

Now, calculate the values:

\[Entropy(S) ≈ 0.4397 + 0.5420\]

\[Entropy(S) ≈ 0.9817\]

So, with 9 "Yes" and 5 "No" instances, the entropy for the entire dataset is approximately 0.9817.














----------------------------------------------------------------------------------------------
from sklearn.tree import plot_tree,export_text
