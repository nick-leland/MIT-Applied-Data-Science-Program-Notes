# Random Forest

We will be focusing here on Statistical Learning, specifically on overfitting.  

Linear regression often works very well because it doesn't amplify the nodes in a strange way.  This results in a model that avoids overfitting very well.  
High Variance estimates in other models is much more of an issue compared to Linear Regression

## Overfitting : Bias-Variance Tradeoff
This is everywhere when working on data.
### Titanic Example
- 891 Data Points
- 38% Survival Rate
Data is split 8:2

Features
- Survived : Indicator variable describing whether the person survived the shikpwreck
This is our predictor
Pclass : Passenger class {1, 2, 3}
Sex : The sex of the passenger
Age : The age of the passenger
Embarked : The port that the passenger embarked from {Cherbourg Southhampton, QAueenstown}
Family Size : the size of the gropu you embarked with.

To build a decision tree, we need to split the model into training and test data.
Lets look at the decision tree we resulted with. It has 6 Levels with an entropy of 0.507
The first featurew we evaluate is whether or not it is a male or female. We have 435 Males and 278 females to base our decision tree on. 

**Train vs Test Results**
We can evaluate the Test Data vs the Training Data and the Depth of the Tree.  
The more you expand a model, the smaller the model.  This is because on the training data we do a better job of overall splitting the data. By increasing the size, we will always reduce the error on the training data.  This will quickly lead to overfitting though! It will fluctuate quite a bit just by changing the number of features.

### Terms used to review the way the data functions
- **Generalization**
- **Extrapolation**

This can be known as the Bias-Variance Tradeoff
How do we handle this? We can increase the model complexity and then when the tradeoff increases drastically, we could select that number of complexity.  
What about testing different types of models? 


### Pruning
Decisiont Trees can be defined to a large level of granularity.  It is not a good way to generalize.
Pruning is when you keep using all of the features that you have until the very end.  We do this until we completely run out of features.  Then, remove branches that do not contribute to an error.  Then you have a smaller tree as a result.

- Use misclassification as guidance
- Reduce the depth
- Eliminate small classes

We create our Decision Tree.  Then we evaluate, "How much do we lose by cutting x branch?".
Every time we cut a branch, we measure how much our training error moves.  There is a tradeoff from a lesser complex graph to a larger training error.  It isn't uniform, we could maintain our overall depth while certain trees get much simpler.  

It is only really appliable to decision trees.  

Steps for Pruning:
1. Create a tree with maximum depth 
  - Either until every leave is a single data point
  - Or use all features
2. Pick a subtree (a node and all leaves)
3. Aggregate that leaves all the way to the node
4. Compute new error
  - Misclassification
  - Entropy

Summary 
- Pruning is expensive (and very generalized)
- Pruining is counter-intuitive 
How about directly building a better model?

Remember, you don't do pruning based on the test samples! Everything needs to happen on the training data.

## Bagging to Reduce Variance
This is part of what is known as Ensemble Learning.
Ensemble learning is the key idea behind Random Forests.
Motivated by averaging techniques. 
You can reduce the variance if you average a number of independent RV's.

We would perform an averaging techique over all of the estimates to determine the overall number.  What is the intuition here?

Maybe we need to ask multiple people to give us models.  Say we have multiple decision trees using the same features over different datasets.  

We only have access to one dataset though.  How do we determine multiple datasets from one? This is known as bagging, we will create the models ourselves from our own data. 

### Ensemble Learning Basic Idea
Bagging = Bootstrap + Aggregation

We use Bootstrap to generate new data from within the dataset. 
Aggregation combines multiple models

We bootstrap, then we aggregate.  Make multiple datasets in a particular way.  Then we build models from these datasets we have.  Then we combine each of those decision trees into one "average" decision tree.  

### Bootstrap 
- Sample data with replacement
  - Create many datasets
- Data not sampled is used for cross validation.  Think of this like our smaller test data.
- Data sets are correlated (or dependent) 
  - Reasonable uncorrelated for large samples

Every time we create a new dataset of size n, there will be some number of indeces that are not included into the dataset.  This will always be different.  We can then used this missing data as cross validation.  

Original Dataset = x1, x2, x3, x4, x5, x6
Bootstrap 1 Dataset = x1, x2, x3, x4, x1, x3  | X5, X6
Bootstrap 2 Dataset = x1, x2, x3, x4, x5, x5  | X6
We now have two new datasets! 
As our number of datapoints become larger, we can determine the percentage of the data that we sampled.  What is the chance that one element doesn't get sampled?
(1/N)
What are the chances we don't pick it every single time? 
(1-(1/N))^N the limit as N reaches infinity is ~0.368%
This essentially says that the datasets are not entirely aligned with each other. 

For each of these datasets, we create a decision tree. Then we can even verify based on the data that is left out!

Build a classifier for each "new" data set
y hat i = fi(x) with error ei
If the n is large, then the outcome of these classifiers are _"not" too dependent_

**Aggregation** is using multiple classiffiers and voting on them (majority vote)
In principle it is possible that you don't do majority but instead you can add weights.  We aren't going to do this for now.  The idea is that this performs better then each individually. 

Yi for each classifier is either 1 or 0. Majority (yi) and then evaluate whether or not it is greater then or less then .5

The exception here is that the Yi's are not all independent of each other.  Here, our bias is based on every bootstrap dataset which could be correlated (our 0.38%).  The idea or hope is that this change is enough to overfit the data in a particular way.  By averaging them we get a better estimate.

Classification Models :    f1  f2  ... fl
                            |   |       |
Predictions  :             y1  y2  ... yl

Voting :                    \  |      /

                           Majority Rule
                             (y hat f) 

{Y hat f} = f(x) = majority(f1(x), f2(x), ..., fl (x))
_You could also combine using the weighted sum_
How does this estimator perform? 

### Analysis of voting
Assume each classifier has error
Errori = P(fi(x) != y hat i) < 0.5

Assume each classifier is independent

How does this error reduce? 

We make an error in the estimate when more then half of the estimator creates an error.  

Say we perform this over 3 different samples
First model says Food Type is the best
Second says how Hungry is the best
Third says that Rain is the best

We then evaluate our data for all 3 points and evaluate the overall best choice.
x1 = 1, 1, 0 -> 1
x2 = 0, 1, 0 -> 0
etc etc.

We would then evaluate the misclassification error for every dataset.
Here maybe we get a 3/12 evaluation.  These models are bad! They are a 1 level each however they together model it all very well.  By averaging, you remove the overfitting.  

The downside is that this is not a tree decision, but an evaluation of multiple trees.  The explanability of this model is lost so it is no longer intuitive compared to the basic decision tree. 

Gain accuracy, lose interpretability 

## Random Forest
What is the difference between random forest and the bagging/ensemble that we currently are using?
The samples that we are using are not independent.  This means that our classifiers are not entirely independent from one another. If each hospital used their own dataset to determine covid likelyhood it **would** be independent, ours were not however. 

You can increase the independence through sampling the features at each node.  

This gives us a **better generalization**.  The downside is that it is less interpretable, less powerful. 

Think of this as one dataset not using a certain feature.  Instead of using all features, we instead sample a random sample of features for each classifier.  Our aggregation techniques can then perform beter.  

Random Forest 
Start with a traning set
Use the bootstrap to create different datasets
Now instead of using all features, we sample a set of features.  Say we have 10 features, we go down to a sample of 6.  
This gives us a number of models based on all of the different outcomes.  
For our prediction we just evaluate the majority vote of our results.

Our added step is the sampling of features

We can go one step further then this.  We can use entropy to determine the diversity of each tree.  There is no fixed level of trees at each point.  
Everything that is added is increasing the randomness which reduced the chance that the errors allign which increases the possible estimates from our problems.  

### Generalization Error of Random Forests
Error <= p bar (1-s^2) / s^2
There are two factors that are playing against each other.  We are sampling the feature space, therefore we are weakening the power of our classifiers, however, we are increasing the power of the aggregation.  

We can captures this tradeoff.  The effect of error is a product based on how correlated the classifiers are to the strenght of the classifier. 

Remember that this tradeoff exists! Effect on Error vs No. of features for Random Forest.

### Classification and Regression Trees (CART)
- What if the features are numerical? (non-binary)
- At each level, you construct a linear classifier of the form : a' x + b >= 0
- Continue the same way!

For each feature, what is the linear curve that seperates the feature to the right or to the left? We want our outcome to be the most homogeneous.  
By making the outcome homogeneous, we are reducing the error percentage. 

The same feature can be run multiple times because we can run teh same regression multiple times.  

Ensemble learning says that we can sample the data for regression with replacement.  Then we can create regression seperation lines over many different regression lines.  

By then performing bagging, we create a fuzziness.  We get a more accurate estimate by not building an accurate model! 

It is interesting because we can utilize simple regressions and then use ensemble learning to gain all of the accuracy back. 

## Summary
- Overfitting increases variance on test data
- Reduce overfitting by 
  - Pruning 
  - Baggin
  - Random Forests
- Regression Trees


