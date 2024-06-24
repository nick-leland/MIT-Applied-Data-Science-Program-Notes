# Weekend Session 2
## Supervised Learning : Classification 

The goals with regression are to predict a continuous variable. 
Classification uses the same workflow as Regression.

A Categorical Label is by definition two or more discrete class labels.

Here, our classification metrics are different. 
Say we are predicting a binary classification (0 or 1)
You would assume that this would have two possible outcomes, however, in reality this has four outcomes due to false positive rates. 
- True Negative
- True Positive 
- False Negative
- False Positive 

Lets look at an example:

Predicted        Actual
           Infected | Healthy
Infected|      0         0
Healthy |      8        92

We have four ways to measure this performance.  
### Accuracy
Accuracy is the proportion of correctly predicted results among the total number of observations.  Think about this as how often the model is correct

Accuracy = (TP + TN ) / Total

This is not always correct in cases where we have an imbalanced dataset. 
**Imbalanced Dataset** : This is a case where the data is not correctly distributed.

In cases like this, we will need to use a different accuracy test.

Precision is the proportion of true positives to all the predicted positives.
Precision = TP / (TP + FP)

F1 Is the harmonic mean of the precision and recall. 

### Logistic Regression
Say we have a dataset with income (-5 -> 5) and Whether or not the loan is payed (0 or 1)
In cases like this, it is not possible to apply linear regression because the output is only 0 or 1. 
Instead of linear regression we can utilize the **Sigmoid Function** which is used to plot Logistic Regression.
This will return the probability that it is in a specific binary class. 

Lets look at an example. We will be testing hearing and using the pysical score and age to determine if they pass a physical test. 
Age | Pysical Score | Test Result

**Reccomended : Standard Scaler of Log Transformation if you have outliers if not Min Max Scaler**

Lets interpret Confusion Matrix
_from sklearn.metrics import confusion_matrix, classificaiton_report, ConfusionMatrixDisplay_

_ConfusionMatrixDisplay.from_estimator(model_lg, scaled_x_test, y_test_
When using the Confusion Matrix, you can also set the normalize value to (true, pred, all)
- True is normalizing over the true conditions
- Pred normalizes over the predicted conditions
- All is normalized by the number of samples

If you print hte classification report, you can view the position recall for each individual class.
Support is the number of samples belonging to a class (Within the test data)

The relationship between the Precision and Recall is inverse.  Select  the threshold value that gives you the best precision/recall tradeoff for your task.  This is usually the interesection point between precision and recall to threshold if you graph it. 

Lets look at an example of KNN.  Say we have heights and weights and we are trying to classify based on gender.  With KNN we classify closest neighbors based on the nearest neighbors.  This is a hyperparameter, we need to select different values and then determine based on error rates.

### Grid Search
This is a way to evaluate HyperParameters that tests multiple different combinations and selects the combination based on the error evaluatoin.
There are two methods 
K-Folds
This is when we divide the training data into different chunks. For each chunk, it moves and trains a different recall value.   This is a very intensive evaluation. 
If you have a very large dataset, this might not be the best evaluation method.
Here, **Holdout** or **Random Search** are probably better options

## Model Building Approach
1. Prepare the data for modeling
2. Partion the data into train and test sets
3. Build the Model on the training data
4. Tune the model if required
5. Test the data on the test set

Really focus on the evaluation with the estimator.


