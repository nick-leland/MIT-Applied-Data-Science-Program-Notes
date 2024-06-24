### Degree of a node
This is the number of edges that are connected to a node.  Say we are looking for celeberties or influencers, these are people that would have high degree counts.
This is a useful graph to visualize the state of the network.  It is also interesting to visualize any limitation factors within networking systems.

### Diameter of a graph
This tells us how quickly news or problems can spread.  
The diameter of a network is hte largest distance between any two nodes in the network. 
_d_ ij denotes the length of the geodesic path (shortest path) between node i and j.
diameter = max _d_ ij (i and j are subscript within V ((our network))) 

### Homophily
Homophily or assortative mixing is the tendency of people to associate with others that are similar.
We can visualize this in a view of a heatmap to view the blocks of association.
These are interesting because essentially we are diving into the social engineering aspects of networks.

## Finding an important Node
**Degree Centrality**
- A high degree centrality.  Think individuals with more connections and more access to information.  
- Does not capture "cascade of effects", importance is better captured by having connections to other important nodes.
- Think, this person is connected to not just a lot of people, but the right people. 

**Eigenvector centrality**
- Score that is proportional to the sum of the score of all neighbors is captured by largest eigenvector of adjacency matrix
- This is essentially the baseline for the PageRank algorithm

**Closeness centrality**  
- Tracks how close a node is to any other node

**Betweenness Centrality**
- This is a way to measure the extent to which a node lies on a path to other nodes.
- High mutuals essentially
- Think about analyzing information for wars, if a node has high betweenness it is a very important target.

What centrality you choose should depend on the application! 

In a social network:
- High degree centrality indicates most popular
- High eigenvector means a popular person friends with other popular people
- High closeness indicates the best spread of information
- High betweenness is the person whose **removal** could best break the network apart.

## Case Study Review
CAVIAR (Criminal network in Montreal)
Network science is often now applied to criminal science as well. 
This data is based off of wiretap warrants from 1994 - 1996 over 11 periods.
This was an interesting situation because the goal was to seize drugs rather then arrests. 
