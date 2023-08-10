# Python Decision Tree

## Description
This is a decision tree using Premier League Table results from 1993 to 2023 to predict table ranking.

## Technologies
+ Python
+ Scikit-learn
  
## What is a decision tree?
Simply put, a decision tree is a machine learning method to classify observations. The resulting tree structure resembles a flowchart with a series of `if-then` statements making it easy to understand and interpret.

![Diagram of Generic Decision Tree](resources/generic_tree.png?raw=true)

## How does a decision tree work?
1. Use an attribute selection measure or ASM to find the best data attribute to split the data. ASM is a heuristic that acts like a shortcut to simplify a problem and provide a solution in a feasible amount of time.

2. Once the `best` attribute is found, it is used to split the data into smaller datasets.

3. Recursively build a `tree` until
  + all the data belongs to the same attribute value
  + there are no more attributes to analyze
  + there is no data remaining

### What determines the ASM?
+ Gini Impurity determines the likelihood that random data would be misclassified. A Gini value is within the interval `[0, 0.5]`. A value of `0` indicates a decision node is `pure` or represents a unique class of data.
+ Entropy determines the disorder or randomness of data. An Entropy value is within the interval `[0, 1]`. The objective is to reduce entropy.

### Example
Build a decision tree to predict the quality of wine using the [Wine Quality Datset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).

#### Result 1
| Command | Prompts | Description |
| ------- | ------- | ---------- |
| ASM method | gini |
| split strategy | best |
| pruning strategy | none |
| Accuracy | 0.5714285714285714 |

![No Pruning](resources/wine_1.png?raw=true)

#### Result 2
| Command | Prompts | Description |
| ------- | ------- | ---------- |
| ASM method | gini |
| split strategy | best |
| pruning strategy | max depth of 3 |
| Accuracy | 0.5830903790087464 |

![No Pruning](resources/wine_2.png?raw=true)

#### Result 3
| Command | Prompts | Description |
| ------- | ------- | ---------- |
| ASM method | gini |
| split strategy | best |
| pruning strategy | max depth of 5 |
| Accuracy | 0.5889212827988338 |

![No Pruning](resources/wine_3.png?raw=true)

#### Result 4
| Command | Prompts | Description |
| ------- | ------- | ---------- |
| ASM method | gini |
| split strategy | best |
 pruning strategy | max depth of 7 |
| Accuracy | 0.6034985422740525 |

![No Pruning](resources/wine_4.png?raw=true)


### References
[Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)