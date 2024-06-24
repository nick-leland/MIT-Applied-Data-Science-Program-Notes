# Weekend Session 1

## Supervised Learning : Regression
The goal is to predict a continuous variable.
The objective simply put is to learn the slope and the intercept of a line. (Our two Thetas)

With Multiple Linear Regression, we are looking for a generalized form of a linear relationship between the features x and the label y.

Keep in mind, OLS (Ordinary Least Squares) are used to find the slop and intercept to minimize the SSE (Sum of Squares Error) for a given dataset.
The Distance from the sample to the line you create is the error.

SSE = Sum( (yi-yhati)^2

What are the limitations of Linear Regression? 
_View the dataset Ansombe's Quartet_
In this dataset, the given line is generated for all four datasets even though the line would not be a good representation. 

We need to make some assumptions and test the assumptions for a given dataset.


### Basic Concepts

### Review of scikit-learn
We first need to split our dataset into two seperate groups, the **Training Set** and the **Test Set**.  In both of these groups you have two categories, **Features** and **Labels**.  We are going to train utilizing the Training Set features and Labels, then we input the Test Set features and compare the actual labels with the predicted labels.

Workflow: 
*Split data into Train and Test*
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=10)
_Here the test size is used to determine the split of training data and test data, random_state is used for seed_
If model overfits you should increase the test size.

*Import Algorithm*
_from sklearn.model_family import ModelAlgo_

*Import Error Metrics*
_from sklearn.metrics import error_metric_

*Create instance of the model* 
_mymodel = ModelAlgo(param1, param2)_

*Train Model*
_mymodel.fit(x_train, y_train)_

*Get predictions*
_predictions = mymodel.predict(X_test)_

*Measure the perforamcne of your model.*
_performance = error_metric(y_test, predictions)_

### Regression Model Evaluation Metrics
We have many different ways to evaluate the performance of a model. 
#### R-Squared
#### Adjusted R-Squared
#### Mean Absolute Error
#### Mean Squared Error
#### Root Mean Square Error
This is the most common because it uses the same units as 'y'

Our goal is to compare predictions with the target using the test set. 

Within scikit learn we can generate multiple of these metrics using SCORERS.
_from sklearn.metrics import SCORERS_

Linear regression does not apply to every individual dataset. 
**Assumptions of Linear Regression**
- There should be a linear relationship between dependent and independent variables.
- No multicollinearity between independent variables (no correlation)
- No Heteroskedascity (Residuals should have a constant variance)
- Residuals must be normally distributed

Use pairplot to view the relationship between different features in a dataset.
We can then evaluate if they woudl be a good fit for Linear Regression.


### Residual plots
For the second assumptions, use a heatmap to view the multicollinearity.
If we have two features with a high correlations between two variables, you need to decide which to variable to utilize. 
This is where you could apply PCA to recude the dimensionality and merge the two. 

We need to evaluate that the residuals (The difference between the true y minus your predicted y hat).  Plot the residual errors against the true y values.
The goal is to have a **random** distribution.
If we can identify a pattern between the residual, Linear Regression is not valid.

How do we calculate residual? 
_test_residuals = y_test - test_predictions_
If you plot the residuals per count, you shouldn't have a skewed distribution.  If you do have  a skewed distribution, you must do feature engineering.
Normal probability plot: 
Compare a 'perfectly' normal distribution against your distribution.


### Bias-Variance Trade-off: Underfitting and Overfitting
We want to have a model that can generalize well to unseen data.
- **bias** is the difference between the prediction of our model and the correct value that we are trying to predict. 
- **variance** : if a model has a high variance, it is clear it pay a lot of attention to the thraining data, including the noise , and does not generalize on the test data. Perform well on training data but have a high error on the test data.

High variance is a clear example of overfitting.
High bias is a clear example of underfitting.
Ideally, we want to have a good balance to achieve a good result.  Low Bias and Low Variance.

## Case Study
### Feature Engineering
This is the process of creating new features through existing features.
Say we have a creation date.  We can then evaluate this to the current date to generate the age of something.

### VIF Scores looks at the multicolinearity between all variables.

