# Weekend Mentor Session 1

## Multiple Testing and their corrections
Example:
If we have two populations with random samples from normal distributions with the same mean and std dev. 

Both distributions are approximately the same.  As the sample sizes increases the average distribution increases. 

## Null vs Alternative Hypothesis
Null is the base that we believe is true
Alternative is what we are trying to prove is not true.

If the P-Value is lower then 5% we reject the null hypothesis.  

FDR vs Bonferroni, there is a link within the google colab that has a good comparison between the two.

This is built into our scipy function, we can utilize 'multipletests'. 

If you are applying a statistical test more then once you apply the multiple tests simply put.  

## Dimensionality Reductions 
The goal is to reduce the complexity of our datasests. This doesn't simply choose a subset, but, it transforms the data to try and maintain as much information as possible. 

### PCA (Principle Component Analysis)
OUtcomes:
- Reduce the number of dimensions in the data
Say we plot the weight and height of a dataset. 
"ansurdf[['weightkg', 'stature']].head()"

Lets add gender to this scatterplot through the hue parameter.

"ansurdf[['weightkg', 'stature']].head()"

sns.scatterplot(data=ansurdf, x='weightkg', y='stature', hue = 'Gender', s=10);

This does a good job at visualizing the two categories, but what if we wanted to visualize 108 features? 

This is where PCA would come in handy. 

Remember, the principle components are a combination of x1 and x2, the original dimensional space.  
We then plot along the new component axis that is generated.

**REMEMBER** if you apply PCA, you will lose information.  It is a necessary loss. 

**Hyperparameters** Are used for necessary parameters when training the model

Example: When working with Linear Regression, our goal is to train a model which predicts the line of best fit.  
If we were to look at polynomial linear regression however, we would need a specific value to train.  This is known as a hyperparameter. 

Every model has hyperparameters that can be adjusted to allow the model to perform better.

In the case of t-SNE, we can adjust the following hyperparameters: 
- Perplexity
- Step (Maximum number of iterations) 
- Epsilon (Learning rate in the range from 10 - 1000) 
You need to test different parameters and see which are the best

Check out the article linked with tSNE on the google colab file

Remember, this is not a one size fits all situation.  Certain dimensionality reduction techniques will work better for different datasets.


## Case Study Air Pollution Data Exploration
We have a dataset with 13 months of data on major pollutants and metrology levels of a city. 

### A good way to view the missing values is wtih the following
df.isnull().sum()
Because True + True evaluates to 2, the sum shows us the total number of missing values. 

How can we check for a high standard deviation? We can compare this directly in a ratio with the standard deviation divided by the mean.

If our values is greater then 0.4=0.7, we know that it is a relatively high standard deviation.

Before we begin, we should plot the variables to identify if the variables skewed, to then observe if we need to apply a transformation to the data. 

**Dummy Variables** are the act of converting string variables to numbers that can be easily input to our model.

Plot the numbers before and after to determien whether or not the method that you chose to fill missing values was worthwhile. 

The objective when working with tSNE is to group the clusters up.  It is a good idea to tune the hyperparameters and plot the results of those. It is important to visualize and use intuition based off of that.

Once you have the tSNE plot, you can then group the invdividual points based on parameters that you set. 

_Look more into enumerate_

_Review the plotting of networks!_
_Research degree centrality more for networks_

