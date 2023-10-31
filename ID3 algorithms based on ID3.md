# Decision tree based ID3 Algorithm
**Decision tree** algorithm is a branch of Machine learning, it is a recursive algorithm which traverses a dataset by keeping a root node and moving onto its branches or child nodes by only considering non-traversed nodes.We can perform regression(numerical prediction) tasks as well as Classification(yes or no prediction) tasks using Decision trees. 
**Why the algorithm is called *decision tree*?**
An analogy between a tree and a decision tree can help illustrate the concept of a decision tree. In a decision tree, you have a root node (analogous to the tree's trunk) that represents the initial decision point. From the root node, there are branches (analogous to the tree's branches) that lead to different decisions or choices. These branches are split based on specific conditions or features related to the data. As you move down the tree, you encounter more nodes and branches, much like the tree's branches and leaves. Ultimately, you reach the leaf nodes (analogous to the tree's leaves), which represent the final outcomes or decisions.
---------------------------------------------------------------------------------------------
### Basic terminologies of Decision trees: 
Decision trees are a popular machine learning algorithm used for both classification and regression tasks. Understanding some basic terminologies associated with decision trees is essential. Here are the key terms:

1. **Root Node:** The topmost node in a decision tree, from which the tree starts branching. It represents the entire dataset.

2. **Internal Node (or Split Node):** A node in the decision tree that has child nodes, meaning it represents a decision point where the data is split based on a particular feature.

3. **Leaf Node (or Terminal Node):** A node in the decision tree that has no child nodes, and it represents the final prediction or decision. In classification, a leaf node corresponds to a class label, and in regression, it contains a numerical value.

4. **Splitting:** The process of dividing the data at an internal node into two or more child nodes based on a feature's value. The feature and its corresponding value that determine the split are called a "splitting criterion."

5. **Splitting Criterion (or Splitting Rule):** The feature and value used to split the data at an internal node. The most common splitting criteria include Gini impurity, entropy, and mean squared error.

6. **Gini Impurity:** A measure of impurity in a node. It quantifies the likelihood of misclassifying a randomly chosen element in the dataset if it were classified according to the class distribution in that node. It is used in classification problems.

7. **Entropy:** A measure of impurity in a node, based on information theory. It quantifies the uncertainty or disorder in a node. Lower entropy indicates higher purity. It is also used in classification problems.

8. **Information Gain:** The reduction in impurity achieved by splitting a node using a particular splitting criterion. Decision trees aim to maximize information gain when choosing the best feature for splitting.

9. **CART (Classification and Regression Trees):** A popular algorithm for decision tree construction, which can be used for both classification and regression tasks. It uses Gini impurity for classification and mean squared error for regression.

10. **Pruning:** The process of reducing the size of a decision tree by removing branches (subtrees) that do not provide significant improvements in prediction. Pruning helps prevent overfitting.

11. **Overfitting:** A situation in which a decision tree model captures noise in the data rather than the underlying patterns, resulting in poor generalization to new, unseen data.

12. **Underfitting:** When a decision tree is too simple to capture the underlying patterns in the data, leading to poor predictive performance.

13. **Maximum Depth:** The maximum number of levels or nodes from the root node to a leaf node in the decision tree. It is a hyperparameter that can be used to control the tree's complexity.

14. **Feature Importance:** A measure of the relevance or importance of each feature in the decision tree for making predictions. It helps identify which features have the most impact on the model's decisions.

These are some of the fundamental terminologies associated with decision trees. Understanding these terms is essential when working with decision tree algorithms and interpreting the results of decision tree models.














----------------------------------------------------------------------------------------------
from sklearn.tree import plot_tree,export_text
