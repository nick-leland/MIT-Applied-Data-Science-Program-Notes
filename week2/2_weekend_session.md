# Unsupervised Learning
Intro to Unsupervised Learning

### Clustering General Concepts
This happens when we are working on unlabeled data, we only have features.  
The goal of this is to group similarities together between groups.  

Determining how we group the clusters is all dependent on the model you are using and the parameters you are observing.  

In some cases, different Data Scientists may percieve certain clusters differently. 

### K-Means Clustering 
This is the easiest to use algorithm.
Two properties you must meet to utilize this algorithm
- Each point must belong to a cluster
- Each point can only belong to **one** cluster

_Review the Visualiing K-Means from colab_

You will test different numbers of centroid (This is a hyperparameter)
The selection of the coordinates centroids are random

The algorithm then computes the distance from each cluster to different datapoints
It takes the average of the coordinates for each cluster and computes a centralized location
This new coordinate is where the updated centroid lives.
After this new centroid is located, we re assign points that are in a certain location.  This repeats several times until there is a convergence.
Inertia has to do with the sum of squares error.

Pros
- Simple to Implement
- Scales to large datasets
- Can easily be adapted to new examplse
- Is guaranteed convergence
Cons


### K-Medoids Clustering
There is a guarantee that the centroids are members of the dataset
It reacts better to outliers when compared to K-Means

The main reason is based on how they determine centroids. 
**K-Means Centroids** are based by calculating the average of the coordinates of all data points.  If a value is VERY high or VERY low, it may affect the mean value

**K-Medoids** updates the new centroid so that the distance between the datapoints and the centroid is minimal. Most of this is due to being forced to use a centroid that is a point within the dataset.



### Expectation Maximization in GMM (Gaussian Mixture Models) clustering
Assumption: The likelihood to belong to each cluster follows a Gaussian distribution.
We fit a set of "K" gaussian distributions to the data.
This means that we can have datapoints that are in between multiple clusters, but they are assigned based on the theoretical distribution.  The algorithm selects this based on the probability it belongs to that cluster. 
Achieves the number of clusters based on the Bayesian Information Criterion (BIC)

**Steps**
- Expectation Step (E-Step) : Initializes the parametesrs and assigns each data point a probability of it belonging to a specific distribution
- Maximization Step (M-Step) : Algorithm updates the parameter estimates by taking the average based on the assigned probabilities from the E-Step

These steps are repeated until convergence is reached.
Convergence is described as falling below a predefined threshold or until a maximum number of iterations is reached.

GMM is similar to K-Means, however, GMM can handle different shapes.  It applies and automatically clusters based on non-circular shapes.  o
It is important to note that we are determining **soft classification** because the result of the clustering is due to the probability it is within that class.

### Hierarchical Clustering
Measures the datapoints that are **most** similar to other data points in order to group in the same cluster. 

**Steps**
- Make each data point a single point cluster, this forms _N_ clusters.
- Take the two closest data points and make them one cluster -> This forms _N-1_ clusters.
- Take the two closest clusters and make them one cluster -> This forms _N-2_ clusters.
- This repeats over and over again until there is only once cluster left.  

This is interesting because we are not determining the amount of clusters to analyze before evaluating.  
I believe it is based on the Euclideqn distance that is set.

The basic concepts: 
We must determine a similarity metric. 
For similaritiy, we can utilize many different ways of calculating distance.
- **Euclidean distance**
- **Manhattan distance**
- Cosine Distance
- Minkowski Distance
- Hamming Distance

Dendogram, this is a visualized distance to display the dataset vs the distance. 

Linkage the the measuring method between different clusters.  It is important to determine how closely clusters may be related to one another.
How to measure linkage: 
- Single Linkage (Minimum distance between two points)
- Complete linkage (Maximium distance)
- Average Linkage (Average distance between every data point)
- Centroid Linkage (Merges the groups whose means are closest.
- Weighted-average linkage (Computes the weighted average of the distance between any data point whose weights are based on the number of elements in the cluster.  Similar to average)
- Ward Linkage (Find the pair of clusters that produces the minimum increase in total variance after we merge them.  This tries to minimize the distance between the datapoints)

Regarding Weighted-Average: 
Weights in this context are based on the number of elements within a cluster. You will be comparing clusters which have different numbers of samples within them so it is important to analyze based on the _value_ that they have within the dataset.

### DBSCAN (Density Based Spatial Clustering of Applications with Noise
This algorithm is not sensitive to outliers and does not require the number of clusters beforehand.
amenities
Utilizes Epsilon and the minumim number of points required to form a dense region

## Case Study
Clustering based algorithms are distance-based algorithms, all distance-based algorithms are affected by the scale of the variables.  If we check the plots, we can see that the values and skews are all different.  This data needs to be normalized.  
We can look at **Standard Scaler** within sciplot. 

In order to determine our Clusters, we need to find the **Elbow Point** where the sum of squares converges. 

### Changing the Scale
How can we change the scale? 
We initialize the aglorithm, transform the data using MinMaxScale
When we scale, the distribution is the same but the data values change.

