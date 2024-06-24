# Exploratory Data Analysis and Visualization

## Week Overview
 - Data collection / Visualization for exploratory data analysis
 - Network analysis
   - Think of recomendation systems or social media algorithms
   - Think of this as advanced 'networks' or systems.
   - Not just tabular data but the connection one instance might have to another (User:User, User:Item, Item:Item)
 - Unsupervised learning (Specifically clustering)
   - Lots of times we don't have specific values and we need to determine those after
   - In an unsupervized setting we need to determine what we are looking for in an unlabeled environment.

### Session 1 Specifics 
 - Data collection: Mammography case study
 - Hypothesis Testing
 - Visualizing high-dimensional data for exploratory data analysis


## Mammography case study
 - Life and death situation, decisions are **very** critical
 - Mammography is the act of screening women for breast cancer, we are interested in determining if this speeds up detection enough to influence the outcomes of cancer for patients. 
 - **How would you approach this problem? What is important when setting up a study / experiment?**
   - Validate our sample from population
   - Random Sampling. The goal is to have the same representation throughout. As long as we randomize control vs treatment, we don't need to worry about the variability of the different individuals. 
    - How do you properly group individuals based on other variables? Think family backgrounds, lifestyle, age, genetics, food, child bearing, etc!
    - Cannot have older women in control group and younger women in the sample group as it could show a false trend.
   - Are there situations where we cannot utilize randomization for sampling?? 
    - Things like Gender cannot be randomized.  In some cases for ethical reasons you cannot randomize
    - Ethical examples: Smoking cannot be forced upon a random group.

### Lets look at the first large-scale randomized controlled experiment on mammography (Performed in 1960's)
    - We are directly looking at deaths within a 5 year followup 
    - Refused are the group of women that decided not to get a mammogram.
    - The cancer rate is similar to the percentage with a **normalized** value.
#### Which numbers should we directly compare to see if mammography influences the results?
    I think that we should directly compare the breast cancer rates between the Screened cancer rate vs Control cancer rate.  We cannot compare the direct numbers because the overall group size are different.  
    - Screened cancer rate vs Refused cancer rate.  There is a decision/selection bias here. This removes the randomization aspect because people were allowed to seperate themselves into categories which inherintly increases the bias. 
    - We are essentially ignoring the refuse group. 
We aren't worried about whether taking a mammogram changes the outcome, its whether **offering mammograms** every year decreases the likelyhood of death by cancer.
If the question was whether people taking a mammogram changed the outcome, people would need to sign that they were accepting the fact they could end up in the group of women who wouldn't have the mammogram outcome (to prevent an ethical issue)
Think of this similarly to showing an advertisement vs the effect of an advertisement.  
Why would the control group have a higher death rate then the treatment group? This cannot just be explained through just education.  In the refused group we could have a lower socioeconomic status.  What is this associated with that is protective of breast cancer? The big difference is child bearing or how many children a person has had. 

## Hypothesis Testing
Now we need to determine if this is just by chance, or if we have enough results that we can determine whether we can effectively determine an outcome.  
Here we only have two outcomes, Survival after 5 years, or Death by Cancer.  What is a good statistical model? **Binomial**.
Now we have a binomial distribution centered around 63 deaths.  Specifically the null distribution. Now we have our treatment group.  P Value is the probability under the null model of the number that we have reached.  With this control model, the probability of 39 deaths is 0.0012.  This is too unlikely to happen by chance, therefore, introducing mammography significantly reduced the results of someone dying through breast cancer. 
This was repeated to ensure that everything was accurate.  

Where are similar real world examples of relevant examples of this? 
- Manufacturing, change in policy or change in technology needs to be able to show a change. 
- Similar to public safety field. 
- Advertising is extremely similar. 

### Significance Level
Lets look at example research findings.

Intake of tomato sauce (p-value of 0.001), tomatoes (p-value of 0.03), and pizza (p-value of 0.05) reduce the risk of prostate cancer;
But for example tomato juice (p-value of 0.67), or cooked spinach (p-value of 0.51), and many other vegetables are not significant

What could be occuring here??
If we looked at the significance, nothing is significant enough to report.  

####
#🐼#
####

Say we create a placebo pill.  
- Randomized group of 1000 people
- Measure 100 variables before and after taking the pill (weight, blood pressure, etc)
- Perform a hypothesis test with a significance level of 5%

What does a significance level of 5% mean? 
It means just by chance, 5 of the 100 variables will be significant just by chance! 

The significance level is much different when looking at different industries.  Think medical vs advertising.  How do we correct for the significance level? Always be careful about studies where many different tests were performed. 
If we have to do a lot of testing, we should have a stricter cutoff.  

**FDR (False Discovery Rate)** Think of just doing screenings.  This is much less strict, generally accepted at less then or equal to 10%.
Expected fration of false significant result among all significant results. 

**FWER(Family wise error rate)** This is strict, the FDA utilizes a value of less then or equal to 5%.
Probability of at least one false significant result.

There are multiple ways to correct for these.  
- Bonferroni Correction (People use this because it is simple to state.
- Holm-Bonferroni Correction (Always use this just because we know have the ease of software)
- Benjamini-Hochberg Correction

You have to accept a certain value of possible mistake.  That is essentially our significance value. 
Always identify specifics with the number of tests being done. 

## What do we do after we have a large dataset?
Lets talk about two very specific approaches.  
- Principal component analysis (Linear)
    Projection that spreads data as much as possible
- Stochastic neighbor embedding (Non-Linear)
    Non-Linear embedding that tries to keep close-by points close.
The price will always be computation time. 

### Principal Component Analysis
**Goal** Dimension reduction to fewer dimensions
**Intuition** Find low-dimensional projection with the largest spread.
This is essentially compressing multi-dimensional data onto a 2 or 3 dimensional space to gather visual intuition.

An example application
We take a set of peoples DNA and by performing principle component analysis we are able to map where someone is from. 

How does PCA work? 
Most importantly **YOU MUST START WITH CENTERED DATA**
Find the dimension where the data varies the most (Largest Variance)
You then plot the data, with the goal of losing as little data as possible.  Mapping the data onto the lower dimensions through linear algebra.
Once you do this, it is much easier to then identify different clusters within the data.

How do we compute this? 
Many of our data science programs compute the eigenvalues and eigenvectors of the matrix. 
How do we get this? It  must compute the eigenvector corresponding to the largest eigenvalue of the covariance matrix to determine the **first** primary dimension.

Look a bit more into the Spectral Decomposition Theorem.
Must be able to quickly determine the covariance matrix and the eigenvectors within those. 

If we change the unit will the first principle component change? 
No because the data must be centered. 
She is saying this does directly change.  This doesn't make sense becuase you are supposed to center and normalize the data prior to performing principle component analysis from my understanding.
She then goes on to confirm that you must center LOL. 

### Stochastic Neighbor embedding (tSNE)
**VERY COMPUTATIONALLY HEAVY**
**Goal** Keep things similar close, but ignore other points.  The goal is to generate clustering.
**Intuition** We would like to keep track of points that are close together to establish potential clustering. We give up points that are not similar to one another.
The goal with a nonlinear system is to place samples from a high dimensional space into a low dimensional space to preserve the identity of 'neighbors'.  
When looking at a dataset of hand written numbers, it is able to cluster the numbers properly together. 

## Resources for Further Learning
- PCA and other Projection Methods : The Elements of Statistical Learning: Data Mining, Inference, and Prediction
- tSNE : Stochastic Neighbor Embedding by NIPS and Visualizing Data using t-SNE by JMLR
