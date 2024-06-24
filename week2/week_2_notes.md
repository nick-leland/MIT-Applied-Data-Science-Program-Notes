# Applied Data Science Class Notes

## 1. Exploratory Data Analysis and Visualization

**Exploratory Data Analysis (EDA)** forms the foundation of any data science project. It involves collecting, visualizing, and understanding data patterns and relationships. Proper data collection methods, including **randomization**, are crucial for ensuring the validity of subsequent analyses.

A prime example of the importance of EDA is the Mammography Case Study. This study aimed to determine if mammography could speed up breast cancer detection enough to influence patient outcomes. When setting up such a critical study, several factors must be considered. These include validating the sample from the population, ensuring random sampling for proper representation, and accounting for variables like family background, lifestyle, age, genetics, and childbearing history. Ethical considerations also play a role, as certain variables (like smoking) cannot be randomly assigned to participants.

The first large-scale **randomized controlled experiment** on mammography in the 1960s provided valuable insights. It examined deaths within a 5-year follow-up period, dividing participants into Screened, Control, and Refused groups. The key comparison was between the screened cancer rate and the control cancer rate. An important realization from this study was that the focus should be on the effect of offering mammograms rather than just taking them, as this accounts for the real-world scenario where some people might refuse the screening.

## 2. Hypothesis Testing and Multiple Testing

**Hypothesis testing** is a fundamental tool in data science for making inferences about populations based on sample data. It involves comparing a **null hypothesis** (what we believe to be true) against an **alternative hypothesis** (what we're trying to prove). Typically, if the **p-value** is lower than 5%, we reject the null hypothesis in favor of the alternative.

However, when multiple tests are performed on the same data, the likelihood of **false positives** increases. This is where **multiple testing corrections** come into play. Two important concepts in this context are:

1. **False Discovery Rate (FDR)**: This is the expected fraction of false significant results among all significant results. It's less strict and generally accepted at ≤ 10%, making it suitable for exploratory analyses.

2. **Family-wise Error Rate (FWER)**: This represents the probability of at least one false significant result. It's stricter, with the FDA typically requiring a value of ≤ 5% for confirmatory analyses.

Several methods exist for correcting multiple tests, including the simple **Bonferroni Correction**, the more powerful **Holm-Bonferroni Correction**, and the **Benjamini-Hochberg Correction**. The choice of method often depends on the specific requirements of the analysis and the field of study.

It's important to note that **significance levels** can vary by industry. Medical research, for instance, typically requires stricter standards than advertising. When encountering studies with many different tests performed, it's crucial to approach the results with caution and consider whether appropriate corrections have been applied.

## 3. Dimensionality Reduction

As datasets grow in complexity, **dimensionality reduction** techniques become essential tools for analysis and visualization. Two prominent methods are **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)**.

**Principal Component Analysis (PCA)** is a linear dimensionality reduction technique. Its goal is to reduce the number of dimensions while retaining as much information as possible. PCA works by finding the dimensions along which the data varies the most, essentially identifying the **principal components** of variation in the dataset.

The PCA process begins with **centered data**. It then finds the dimension where data varies the most (the largest variance) and projects the data onto this new dimension. This process is repeated for subsequent dimensions, always choosing the direction of maximum remaining variance. Computationally, this involves calculating the **eigenvalues** and **eigenvectors** of the data's **covariance matrix**.

An important consideration in PCA is the effect of data scaling. Changing units can affect PCA results if the variables are not standardized. While **centering** the data (subtracting the mean) is crucial, it alone doesn't make PCA invariant to units. For this reason, it's standard practice to both center and **scale** the data (**standardization**) before applying PCA, which does make it invariant to the original units.

In contrast, **t-SNE** is a non-linear dimensionality reduction technique. Its goal is to keep similar data points close together in the lower-dimensional space while allowing dissimilar points to be far apart. This makes t-SNE particularly effective at revealing clusters and local structures in complex datasets, such as collections of handwritten digits.

While t-SNE is powerful, it's also computationally intensive, especially for large datasets. It's often used as a visualization tool rather than for general dimensionality reduction.

The choice between PCA and t-SNE (or other techniques) depends on the specific goals of the analysis, the nature of the dataset, and the available computational resources. PCA is often a good starting point due to its simplicity and efficiency, while t-SNE can provide more nuanced insights into local data structures.

## 4. Network Analysis

**Network analysis** is a powerful approach for understanding complex systems of interconnected entities. In this context, a **network** or **graph** (G) is defined as a collection of **nodes** or **vertices** (V) connected by **links** or **edges** (E), denoted as G = (V, E).

Networks can represent a wide variety of systems. In social networks, nodes represent people and edges represent friendships or other relationships. In transportation systems like subway networks, nodes might be stations connected by tracks (edges). Financial systems can be represented as networks too, with stocks as nodes and their correlations as edges.

Several types of networks exist, each with unique properties:

- **Simple Networks** have no self-loops (nodes connecting to themselves).
- **Multigraphs** allow self-loops and multiple links between vertices.
- **Directed Networks** have edges with specific directions.
- **Weighted Networks** assign attributes to edges or vertices.
- **Bipartite Networks** have edges between two distinct sets of nodes, but not within each set.
- **Hypergraphs** allow interactions between more than two nodes.

Networks can be represented in various ways, with two common methods being **adjacency matrices** and **adjacency lists**. An adjacency matrix is an nxn matrix (where n is the number of nodes) that can represent binary connections or weighted edges. Adjacency lists, on the other hand, only keep track of existing edges, making them more efficient for sparse networks.

Several measures help us understand network properties:

1. **Degree of a node**: This is simply the number of edges connected to a node. In social networks, high-degree nodes might represent influential individuals.

2. **Diameter of a graph**: This is the largest distance between any two nodes in the network, indicating how quickly information or effects can spread across the network.

3. **Homophily**: This refers to the tendency of similar nodes to connect, often visualized using heatmaps.

**Centrality measures** are particularly useful for identifying important nodes:

- **Degree Centrality** measures the number of connections a node has.
- **Eigenvector Centrality** considers not just the number of connections, but the importance of those connections.
- **Closeness Centrality** measures how close a node is to all other nodes.
- **Betweenness Centrality** measures the extent to which a node lies on paths between other nodes.

The choice of centrality measure depends on the specific application. In a social network, for instance, high degree centrality might indicate popularity, while high betweenness centrality might identify individuals crucial for information flow.

## 5. Unsupervised Learning

**Unsupervised learning** is a branch of machine learning that works with unlabeled data, aiming to discover patterns and structures within the dataset. One of the primary tasks in unsupervised learning is **clustering**, which groups similar data points together.

Several clustering algorithms are commonly used, each with its own strengths and characteristics:

**K-Means Clustering** is one of the simplest and most widely used algorithms. It divides the data into K clusters, where K is predetermined. The algorithm works by randomly placing K centroids, assigning each data point to the nearest centroid, then recalculating the centroid positions based on the assigned points. This process repeats until convergence. K-Means is simple to implement and scales well to large datasets, but it can be sensitive to the initial placement of centroids and may not work well with non-circular clusters.

**K-Medoids Clustering** is similar to K-Means, but with one key difference: the centroids are guaranteed to be members of the dataset. This makes K-Medoids more robust to outliers compared to K-Means. The centroid update step in K-Medoids minimizes the sum of distances between datapoints and the centroid, rather than taking the mean position.

**Gaussian Mixture Models (GMM)** assume that the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. GMM uses an iterative process of **Expectation-Maximization (EM)** to fit these distributions to the data. This approach allows for **soft clustering**, where each point has a probability of belonging to each cluster, making it more flexible than hard clustering methods like K-Means.

**Hierarchical Clustering** builds a hierarchy of clusters, either through an **agglomerative** (bottom-up) or **divisive** (top-down) approach. In the agglomerative approach, each data point starts as its own cluster, and the algorithm progressively merges the closest clusters until only one remains. This method doesn't require specifying the number of clusters beforehand and provides a **dendrogram** visualization of the clustering process. However, the choice of **distance metric** and **linkage method** can significantly affect the results.

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is particularly useful for datasets with clusters of varying shapes and sizes. It defines clusters as dense regions separated by sparser regions, using two main parameters: **epsilon** (the radius of neighborhood) and the minimum number of points required to form a dense region. DBSCAN can identify clusters of arbitrary shape and is robust to outliers, which it labels as noise.

When applying clustering algorithms, it's crucial to **preprocess** the data appropriately. This often involves **normalizing** the data, especially for distance-based algorithms, to ensure that all features contribute equally to the clustering. Techniques like the **elbow method** can help determine the optimal number of clusters for algorithms that require this parameter.

The choice of clustering method depends on the specific characteristics of your dataset and the goals of your analysis. It's often beneficial to try multiple methods and compare their results to gain a comprehensive understanding of the structures within your data.

## 6. Case Studies

Case studies provide valuable opportunities to apply data science concepts to real-world scenarios. Here are a few examples from our course:

The Mammography Study, which we touched upon earlier, analyzed the impact of offering mammograms on breast cancer mortality. This study highlighted the importance of proper randomization in experimental design and the need to consider **selection bias**, especially when dealing with a "refused" group in medical studies.

In the Air Pollution Data Exploration case, we worked with 13 months of data on major pollutants and meteorology levels. This study walked us through several key steps in data analysis:

1. We began by checking for missing values, a crucial step in ensuring data quality.
2. We analyzed the **standard deviation** of our variables, considering a ratio of standard deviation to mean greater than 0.4-0.7 as indicative of high variability.
3. We plotted variables to identify any **skewness**, considering transformations for highly skewed data to improve the performance of our subsequent analyses.
4. We used **dummy variables** to encode categorical data, making it suitable for numerical analysis.
5. We visualized our data before and after filling missing values, helping us understand the impact of our **imputation methods**.
6. We applied dimensionality reduction techniques, particularly t-SNE, and learned the importance of tuning **hyperparameters** for optimal results.
7. Finally, we grouped individual data points based on set parameters, allowing us to identify patterns and clusters in our air pollution data.

The CAVIAR (Criminal Network in Montreal) case study demonstrated the application of network science to criminal investigations. Based on wiretap warrants from 1994-1996, this study aimed to understand and disrupt drug trafficking networks. Unlike traditional law enforcement approaches that focus on arrests, the goal here was to seize drugs, showcasing how network analysis can inform strategic decision-making in complex scenarios.

These case studies underscore the diverse applications of data science techniques, from healthcare and environmental studies to law enforcement. They also highlight the importance of adapting our analytical approaches to the specific context and goals of each unique situation.

## Key Terms for Flashcards

1. Exploratory Data Analysis (EDA): A statistical approach to analyzing datasets to summarize their main characteristics, often with visual methods. It's used to understand data patterns, spot anomalies, test hypotheses, and check assumptions.

2. Randomization: The process of assigning participants in an experiment to different groups (e.g., treatment and control) by chance, to reduce bias and ensure statistical validity.

3. Randomized controlled experiment: A type of scientific experiment where participants are randomly allocated to either the experimental group (receiving the treatment) or the control group. It's considered the gold standard for determining causality.

4. Hypothesis testing: A statistical method used to make inferences about a population parameter based on a sample statistic. It involves comparing a null hypothesis against an alternative hypothesis.

5. Null hypothesis: A statement of no effect or no difference, typically the hypothesis that the researcher tries to disprove.

6. Alternative hypothesis: A statement of some effect or difference, typically the hypothesis that the researcher hopes to support.

7. P-value: The probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis is true. A small p-value (typically ≤ 0.05) suggests strong evidence against the null hypothesis.

8. False positives: Also known as Type I errors, these occur when a test incorrectly rejects a true null hypothesis.

9. Multiple testing corrections: Statistical methods used to correct for the increased chance of false positives when performing multiple hypothesis tests on the same dataset.

10. False Discovery Rate (FDR): The expected proportion of false positives among all significant results. It's less stringent than the family-wise error rate and is often used in exploratory research.

11. Family-wise Error Rate (FWER): The probability of making one or more false discoveries, or type I errors, when performing multiple hypotheses tests.

12. Bonferroni Correction: A simple multiple-comparison correction used to control the family-wise error rate. It adjusts the significance level for each test by dividing it by the number of tests performed.

13. Holm-Bonferroni Correction: A sequential method for controlling the family-wise error rate that is more powerful than the standard Bonferroni correction.

14. Benjamini-Hochberg Correction: A method to control the false discovery rate in multiple comparisons. It's less conservative than methods controlling the family-wise error rate, making it useful in exploratory analyses.

15. Significance levels: The threshold value that determines whether a test result is considered statistically significant. Common levels include 0.05 and 0.01.

16. Dimensionality reduction: Techniques used to reduce the number of features in a dataset while retaining as much of the important information as possible. This can help with visualization and can mitigate the "curse of dimensionality."

17. Principal Component Analysis (PCA): A linear dimensionality reduction technique that transforms the data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they explain in the data.

18. t-Distributed Stochastic Neighbor Embedding (t-SNE): A non-linear dimensionality reduction technique particularly well suited for visualizing high-dimensional data. It aims to preserve local structure, making it good at revealing clusters.

19. Principal components: The directions in feature space along which the data varies the most. In PCA, these are the eigenvectors of the data's covariance matrix.

20. Centered data: Data that has been transformed so that its mean is zero. This is typically done by subtracting the mean value from each data point.

21. Eigenvalues: In the context of PCA, eigenvalues represent the amount of variance explained by each principal component.

22. Eigenvectors: In the context of PCA, eigenvectors represent the directions of the principal components.

23. Covariance matrix: A square matrix giving the covariance between each pair of elements in a dataset. It's used in PCA to find the directions of maximum variance.

24. Centering: The process of subtracting the mean from each feature, resulting in centered data.

25. Scaling: The process of dividing each feature by its standard deviation (or another measure of spread) to ensure all features are on a similar scale.

26. Standardization: The process of both centering and scaling data, typically resulting in features with zero mean and unit variance.

27. Network analysis: The study of graphs as a representation of relations between discrete objects. It has applications in many fields, including social network analysis, biology, and computer science.

28. Network: A representation of a set of objects (nodes) where some pairs of objects are connected by links (edges).

29. Graph: In mathematics, a structure consisting of a set of objects (vertices or nodes) that are connected by edges or arcs. It's the mathematical foundation for network analysis.

30. Nodes: The fundamental units of which graphs are formed. They represent the objects in a network.

31. Vertices: Another term for nodes in a graph or network.

32. Links: The connections between nodes in a network.

33. Edges: Another term for links in a graph or network.

34. Simple Networks: Networks where there is at most one edge between any two nodes and no self-loops (edges connecting a node to itself).

35. Multigraphs: Graphs that are allowed to have multiple edges between the same pair of vertices.

36. Directed Networks: Networks where edges have a direction, pointing from one node to another.

37. Weighted Networks: Networks where edges have a numerical value (weight) associated with them.

38. Bipartite Networks: Networks whose nodes can be divided into two disjoint sets such that every edge connects a node in one set to a node in the other set.

39. Hypergraphs: A generalization of a graph in which an edge can connect any number of vertices.

40. Adjacency matrices: A square matrix used to represent a finite graph. The elements of the matrix indicate whether pairs of vertices are adjacent or not in the graph.

41. Adjacency lists: A collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a vertex in the graph.

42. Degree of a node: The number of edges connected to a node.

43. Diameter of a graph: The maximum distance between any pair of vertices in the graph.

44. Homophily: The tendency of individuals to associate and bond with similar others.

45. Centrality measures: Indicators of the importance of nodes within a network.

46. Degree Centrality: A measure of node importance based on the number of connections it has.

47. Eigenvector Centrality: A measure of node importance that considers not just the number of connections, but also the importance of those connections.

48. Closeness Centrality: A measure of how close a node is to all other nodes in the network.

49. Betweenness Centrality: A measure of how often a node lies on the shortest path between other nodes in the network.

50. Unsupervised learning: A type of machine learning where the algorithm is given input data without explicit instructions on what to do with it. The goal is to model the underlying structure or distribution in the data.

51. Clustering: The task of grouping a set of objects in such a way that objects in the same group (cluster) are more similar to each other than to those in other groups.

52. K-Means Clustering: A method of vector quantization that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster center).

53. K-Medoids Clustering: A clustering algorithm related to k-means clustering that uses actual points in the dataset as the cluster centers instead of the mean.

54. Gaussian Mixture Models (GMM): A probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

55. Expectation-Maximization (EM): An iterative method to find maximum likelihood estimates of parameters in statistical models, where the model depends on unobserved latent variables.

56. Soft clustering: A clustering method where each data point is assigned a probability or likelihood of belonging to each cluster, rather than being assigned to a single cluster.

57. Hierarchical Clustering: A method of cluster analysis which seeks to build a hierarchy of clusters. It can be agglomerative (bottom-up) or divisive (top-down).

58. Agglomerative: A "bottom-up" approach in hierarchical clustering: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

59. Divisive: A "top-down" approach in hierarchical clustering: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

60. Dendrogram: A tree diagram used to illustrate the arrangement of the clusters produced by hierarchical clustering.

61. Distance metric: A function that defines a distance between each pair of elements in a set. Common examples include Euclidean distance and Manhattan distance.

62. Linkage method: In hierarchical clustering, the method used to calculate the distance between clusters. Common methods include single linkage, complete linkage, and average linkage.

63. DBSCAN: Density-Based Spatial Clustering of Applications with Noise. A density-based clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.

64. Epsilon: In DBSCAN, the maximum distance between two samples for one to be considered as in the neighborhood of the other.

65. Preprocess: The process of transforming raw data into a format that will be more easily and effectively processed for the purpose of the user.

66. Normalizing: The process of scaling individual samples to have unit norm. This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.

67. Elbow method: A method used to determine the optimal number of clusters in k-means clustering. It plots the explained variation as a function of the number of clusters and picks the elbow of the curve as the optimal number of clusters.

68. Selection bias: The bias introduced by the selection of individuals, groups or data for analysis in such a way that proper randomization is not achieved, thereby ensuring that the sample obtained is not representative of the population intended to be analyzed.

69. Standard deviation: A measure of the amount of variation or dispersion of a set of values. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.

70. Skewness: A measure of the asymmetry of the probability distribution of a real-valued random variable about its mean.

71. Dummy variables: Also known as indicator variables, these are binary (0 or 1) variables used to represent categorical data in statistical analysis and machine learning models.

72. Imputation methods: Techniques used to replace missing data with substituted values. Common methods include mean imputation, regression imputation, and multiple imputation.

73. Hyperparameters: Parameters whose values are set before the learning process begins. These parameters are not learned from the data, but are used to control the learning process itself.


