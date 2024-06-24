# Statistics for Data Science

## Descriptive vs Inferential Statistics
We utilize statistics to give a snapshot of numerical dataset.  
What does this data tell us about something more? 

We are looking historically but the goal is to find out information about the future.  

Inferential statistics allow us to answer questions that reach further then the information we have within our dataset.

What is the chance a baby's birth weight is < 3000 grams?
Descriptive answer : Count the births < 3000 grams
Inferential Answer : Consider the underlying distribution and calculate the proportional area < 3000 grams. 

The inferential answer gives us something that we can use not just on this dataset, but on further datasets. 

People do not like being predictable that is a charm of life and the charm of life extends to challenges in business. 

Example business problems to utilize statistics:
Is the new manufaturing process better/more reliable than the odl process

## Fundamental Terms in Distributions
Random Variable : Something that tries to measure the difference between an unpredictable uncertainty. 
A random variable assigns a numerical value to each outcome of an experiment.  It assumes different values with different probability.

Discrete Random Variable : A random variable taht can only take on a countable number of distinct values.  The probabilies are assigned to teh different countable values.

Continuous Random Variable : Think about a float variable where we are observing the percentage left in a cup.  This is something that cannot be categorized based on the amount left, because it can always be to a further precision value. 

Probability Distributions can be either Discrete Probability Distribution or a Continuous Probability Distribution.  
Discrete is a direct value 
Continuous is a Region

Lets look at an example.  If a company tracks the number of new employees and tracks the progress during an 100 day period, say we have an employee that has a certain number of sales among a certain number of days. 
We can create a distribution by looking at the relative frequency which is a probability that we are assigning to the chance of a sale.

This is looking at one employee directly but it could apply to other employees (Think inferential vs direct)

## Distributions around us
Bernoulli distribution (Think of this as a binary distribution)
Binomial distribution, each individual 

## Binomial Distributions
This is a discrete distribution, with two outcomes, 1 (success) or 0 (failure)
Important to note a success doesn't have to be a positive thing (Non-judgemental)

Bernoulli Distribution is a special case of the binomial distribution.

x is either
1, with prob p
0, wth prob 1-p

Think manufacturing defective parts or the outcome of a medical test.

If an answer is Yes or No and the event is random, we can use the bernoulli distribution to model this scenario.  
We can define a random variable X which counts the number of successes (Either the number of adults that responded yes or no)

### Probability mass function
Requires independent trials
The trials (n)  must be fixed
There are only two possible outcomes (Success or Failure) ((0 or 1))
The probability of success (p) is the same for each trial

If these cases are not met, the assumptions will begin to be approximate but this might be good enough for practical purposes.

## Uniform Distribution
If all of the outcomes have an equal probability of occurrence and are mutually exclusive the probabilities of occurrence are uniformly distributed.

There are two version of Uniform Distribution
1. Discrete Uniform Distribution - Where we can count the outcomes. Can take finite number (m) of values and each has equal probability of selection.
2. Continuous Uniform Distribution - Where we cannot count the outcomes.  Can take any value between a specified range. 

## Normal Distribution
Think of a symetric bell-shaped curve
Has two specific parameters, mean (mew) and standard deviation (sigma). This typically refferes to the populations distribution NOT the sample.

Normal distribution due to it being commonly found everywhere. 
### Normal Distribution Properties
1. Symetric around the mean
2. Mean, Median and Mode of the normal distribution are all equal
3. Total area under the normal curve is 1

#### Empirical Rule
4. About 68% of the data fall within one standard deviation from the mean
5. About 95% of the data fall within two standard deviations from the mean
6. About 99.7% of the data fall within three standard deviation from the mean.

You can utilize this to essentailly call out a 'confidence factor' in the data that you are covering.
In order to do this you will need to determine the area under the density curve.
We will need calculus to determine this area.

The standard normal varialbe is denoted by Z and the distribution is also known as Z distribution
It always has a mean of 0 and a standard deviation of 1.  This is typically reffered to as standardizing data (I believe)
We can evaluate this through the Z Score

To convert a normal variable to a standard normal variable, subtract the mean and divide the standard deviation.  
