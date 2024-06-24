# Decision Trees & Random Forest
This starts the second portion of the course.  

## Decision Trees
**Root Node** : the very first condition that the algorithm checks for.  After that  there are **Leaf (Terminal) Nodes** where we further split the data.  Inside this tree, there are more **subsets** of the tree (comprised of other root nodes and leaves)

Inpurity is the mathematical measurement of how pure (homogeneous) the information in our dataset is.  

How can we evaluate this? 

### Gini impurity
_This is important make sure to study this_
For a given dataset _Q_ and a set of classes _C_ : 

G(Q) = sum(Pc(1-pc)) 
Pc = 1/Nq * sum(1(yclass = c))

Lets say we are working with a binary classification.  
In our decision tree, we have the following:
Root Node : X - Calories Intake (Per day) [High / Low]

        No : 2
          /
       High -> Yes : 3
        /
Calorie Intake
        \
        Low -> Yes : 1
         \
        No : 3

In an example, we would split our dataset into two categories:
High (Yes : 3, No : 2)
Low (Yes : 1, No : 3)

What is the probability that in the high class we have a probability of a true case? 
3/5 (3 Positives and a total of 5 variables)

Gleft = (3/5) * (1-(3/5)) + (2/5) * (1-(2/5)) 
Gleft = 0.48

Gright = (3/4) * (1-(3/4)) + (1/4) * (1-(1/4))
Gright = 0.375

We can use the weighted average to calculate the Gini impourity of features
_remember to review weighted averages_

G = 0.48 (Gleft) * (5/9) + 0.375 (Gright) * (4/9) = 0.4333

If all of the sampels belong to the same class, the data is homogeneous

We use this because the algorithm is going to apply this formula and determine the feature with the losest impurity. 

There are many different versions here:
The algorithm will determine the best metric
### Gin Index
- Easy to compute 
- Non additive
### Entropy
- Computationally Intensive
- Additive
### Information Gain
- Computationally Intensive
### Variance
- The most common when dispersion .... (scrolled)

### Decision trees can be more complex. 
- We can use multiple features
- We can use continuous features alongside discrete features
- Multi-categorical features

### Advantages of Decision Trees
Pros 
- Simple to understand and interpret
- Easy to use
- Versatile
- Powerful
Disadvantages
- No guarantee of using all features
- Rood node always be the same 
- The root node has a **Huge** influence over the tree
- Still prone to overfitting

## Ensemble Learning
You will often get better predictions with aggregate groups of predictors over a signle individual predictor.  A group of predictors is called an **ensemble** therefore we can call this **ensemble learning**.

If we trained a few different classifiers with the same dataset, we can use ensemble learning to determine what has a higher accuracy then the best class in the ensemble

The objective of ensemble learning is to create either a hard voting classifier or a soft voting classifier.
**Hard Voting**: Aggregate the predictions of each classifier and predict the class that gets the most votes
**Soft Voting**: This determines 


## Random Forest
This is an ensemble of a bunch of diferent decision trees

What is occuring? Say we have a dataset with 5 different features, and then a target.  
The first step is to randomly select 'n' number of features.  The number of features is a direct hyperparameter. 
Say in this case we select 3 different features 
We then also select a number of subsets (Another hyperparameter) 
In this case we will select 3 different subsets. 

In essence, the algorithm selects n number of subsets and m number of features (n, m both hyperparameters)
We will create different decision trees for each set of algorithm within here.  Because of this, the shape of each tree is different as the algorithm is going to use different features and then of course, there will be different selections of Gini Impurities.

Say we are left with 5 Trees.  
Tree 1: Y=0
Tree 2: Y=1
Tree 3: Y=0
Tree 4: Y=0
Tree 5: Y=0

Our goal is to join or _ensemble_ the decision of all of the trees within the forest.

The outcome of this forest is 0 with 80% (5 Trees, 4 with the same classification{0})

### Hyperparameters
These are off of the randomforestclassifier within scikit learn
Remember, with hyperparameters, we don't need the best combination, this requires trial and error.
_n_estimators_ = The number of trees within your forest.
_bootstrap_ Is whether or not we allow for bootstrapping
_oob_score_ = Should we calculate "Out of Bag Error"
_max_features_ : This is the total number we are looking at.  We can use an integer (total number), float (a fraction of total features), None (_max_features_ is equal to _n_features_)
_n_jobs_ is the maximum number of concurrently running workers. (1 (no joblib parallelism), -1 (all CPU's are used), n (n number of cores are used)
Verbose controls the verbosity when fitting and predicting _Look more into this_
_warm_start_ When set to True, we reuse the solution of the previous call to fit and add estimators to the ensemble.  Realisticlly this will not actually improve the results very much. 

Use between 50 and 250 trees for the most part. 

## Bootstrap Aggregation 
Bootstrapping is used to allow duplicate observations within the same subset. 
A term to describe the process of random sampling with replacement

If you apply bootstrapping, you allow repetition. 

The main idea here is to **reduce correlation between the trees**

### Bagging within Random Forest
Merge the outputs of various models to get a final result
This reduces the chances of overfitting and allows training process to be done in parrallel.

### OOB Score
If we perform bootstrapping, the model is not going to use all rows during training (Replacing with duplicates)
You can use the remaining data (OOB or **Out of Bag** samples)

Dataset: 
Patient A
Patient B
Patient C
Patient D

Bootstrap Sample: 
Patient A
Patient A
Patient C
Patient C

Out-of-bag Set:
Patient B
Patient D

Then we would use the Out-of-bag sample to evaluate the training data.

If we use this approach, we wouldn't need to randomly split the data between train and test (random forest will do this for you). 
You can still do this (and it is preffered) but it is up to you. 

## Advantages and Disadvantages of Random Forest
Pros: 
- Can be used to solve classification and regression
- One of the most accurate algorithms due to the number of decision trees taking part in the process
- in general (not always), does not suffer from overfitting
- Used to select features of relatively more importance within a decision tree.
Cons: 


## What is the best algorithm?
There is no way to know, you must test everytrhing.  With Scikit, the workflow is overall the same. Hyperparameters are different and you must know the documentation on how to manipualte the values but overall it is trial and error.

# Case Study
## Objectives of Exploratory Data Analysis
- Looking for correlations within the data
- Variate and Multivariate analysis
- Missing Data and Outliers

### Model Building Approach
1. Data preperation
2. Partion the data into train and test
3. Build a model on the training data
4. Tune the model if required
5. Test the data on the test set

## Data Preperation
- Create dummy variables (String to numerical) 
  - Use _.drop_first=True_ when doing this for ease of use.
- Seperate the indipendent varables(X) and dependent variables (y)
- Split the dataset

## Split the data into Train and Test
y = df."Y-Axis-Here"
X = df of everything without head. 

_X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, stratify=Y)_
Here we use a 70% / 30% split.
You **NEED** to use stratify if the dataset is imbalanced.
Stratify will weight the class (I believe, double check this)

## Model Evaluation Criterion
Model can make two types of mistakes (False Positive, False Negative)

True Positive  | False Positive
False Negative | True Negative

Precision = True Positive / Actual Result
Recall = True Positive / Predicted Results 
Accuracy = ( True Positive + True Negative ) / Total

What case is more important? It overall depends on the situation and depending on the case that we are evaluating. 
There is always a trade off here.

We will evaluate by using a _classification_report_ to generate the report that will give precision, recall and f1-score. 
f1-Score is the harmonic mean of precision and recall.  You can think of this as _overall_ score.

## Building the Model

Here we will be building a Decision Tree and a Random FOrest Model.  

In order to evaluate an imbalanced dataset, we will have to assign weights for each class.
Random forest and decision tree both allow us to do this.  

We can use the information of the countplot to assign the weights

_dt_weights_ = DecisionTreeClassifier(_class_weights_ = {0: 0.17, 1: 0.83}, _random_state_=1
Weights are **different** from stratify.  We use stratify when splitting the data between train and test. 

## Evaluate the model
Perform model evaluation by reviewing the countplot.

In order to see the importance of the values that we have, we can use feature importance

importances = _dt_weights.feature_importances_
This is a series with the features and the overall importance metric.

Now we can rename the index based on feature columns
_importance_df_ = pd.DataFrame(importances, index=columns, columns = ['Importance'])

Where are these values coming from? The importance calculation.  

We can also plot this to view this visually.

We can then optimize using our hyperparameters.

We can do this by creating an array with the combinations that we would like to test.
We will do this by creating a **DecisionTreeClassifier**

Then we can make a parameters dictionary with what parameters we can adjust
parameters = {'_max_depth_": np.arrange(2, 7), 'criterion':['gini', 'entropy'], '_min_samples_leaf_': [5, 10, 20, 25]}

Use a wide range at the beginning and slowly adjust the parameters until the best values converge. 
This is just to reduce overall computation time.
Similar to cross-validation

You can then find the best values from this point.

You can plot the decision tree with _tree.plot_tree_

The workflow is exactly the same when working with randomforest vs decision trees

If a decision tree is overfitting the training set, is it a good idea to decrease max depth?


If a decision tree is underfitting the training set, is it a good idea to scale the input features?
There is no difference if you change the scale for categorical features.

Is it necessary to fill missing values to use the Decision Trees and Random Forest Algorithm?

