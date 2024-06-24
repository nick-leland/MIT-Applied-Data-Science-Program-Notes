# Explatory Data Analysis and Visualization

## Session 2 Specifics
- Examples of networks and representing networks
  - How samples are linked with one another (Friends on a social network)
- Summary statistics of a network
- Centrality measures (Important nodes within a network)

## Network
A **network** or **graph** (G) is a collection of nodes or verticies (v) connected by links or edges (E).  
We can denote this as G = (V, E). 
_Does this also match algorithms?_
This is a relatively new thing as computation power wasn't as readily avalible.  Now we have the capabilities to work directly with **large** networks which is extremely important.
Think facebook friend reccomendation (This is trying to detect which nodes are missing within a network)

### Examples
Social Network
- Nodes People
- Edges are Friendship/Following

Subway Station
- Nodes are Trains
- Edges are the Tracks

Amazon Reccomender System
- Individuals as Nodes / Products as Nodes
- Edges are if an individual has purchased or reviewed the item

Stock Market
- Stocks are Nodes
- Edges are stock correlation

Sports
- Teams are Nodes
- Whether the team has played is an edge

### Kinds of Networks
**Simple Network**: Typical network with nodes and edges. No self-loops (Cannot connect to itself)
- Internet, power grid, telephone network
**Multigraph**:self-loops and multiple links between vertices are possible
- neural network, road network
**Directed Network**: i -> j does not imply j -> i
- World Wide Web, food web, citation network
**Weighted Network**: with edge weights or vertex attributes
- Transportation Networks
**Bipartite Network**: Edges between but not within
- Recomendation System
**Hypergraph**: generalized 'edges' for interactions between > 2 nodes.
- Protein protein interaction network, _Look into other examples_

How do we represent a network visualized??
Big networks look like hairballs.
How can we represent this in a different way? 
How can we represent it on a computer?

**Adjacency Matrix**
This is a matrix that has rows and columns for each node.  It is binary based on whether the nodes are connected.
Of size nxn (where n is the amount of nodes)

Chart   x---x---x
Node    1   2   3

   1 2 3
1 [0 1 0]
2 [1 0 1]
3 [0 1 0]

We can also represent this with weights. 
Chart   x---x---x
Node    1   2   3
Weights 1   2  -1

We can represent this chart with weights as follows
    1  2  3
1 [ 0  1  0]
2 [ 1  0  2]
3 [ 0  2 -1]

What happens if we are looking at a network with direction? (Directed Network)
Chart   x-->x<--x
Node    1   2   3
    1  2  3
1 [ 0  1  0]
2 [ 0  0  0]
3 [ 0  1  0]

### Adjacency List
These matrixes are difficult when it comes to scale.  We don't really care about the 0 entries, which is the benefit of an adjacency list.
**Adjacency List** only keeps track of the edges.

Representation of a network G = (V, E)
- We can represent the first list we made (Undirected) as follows:
E = {{1, 2}, {2, 3}}
- The second list (Directed) can be represented as follows:
E = {(1, 2), (3, 2)}
If we were adding weights, we would add another entry that would correspond to the weights.
for instance, {{1, 2}, {2, 3}} would turn into {{1, 2, 1}, {2, 3, 2}}

Say we have a big network.  How do we go about finding missing links? How do we propose a predicted network trend that one person hsould be linked with another? 

How do we apply the adjacency matrix to a larger scale network? 
    1  2  3  4  5  6
1 [ 0  1  0  0  1  0]
2 [ 0  0  1  0  1  0]

Lets say we are looking at a social network where each node is a person.  The 1's represent if a node is connected to another node (friends or following etc).
How can we evaluate the adjacency list here? Simply the **inner product** of the matrix looking at both users! 

A x A is the number of common friends
where A is a symetric matrix (Therefore not necessary to use A^T)

A x A is used to determine the lengths of paths 2.  
What if we wanted to determine the paths of length 3? We would utilize A^3 or A x A x A. 
This is the basic essence of network science.  

## Quantitative measures of networks
- Connected components
- degeree distribution
- diameter and average path length
- homophily or assortatitve mixing

#### Connected Components
Simply put, this is a set of nodes that are reachable from one another but not connected to other sets of nodes. 
Many networks consit of one large component and many smaller ones. 
Component size distribution in the 2011 Facebook network on a log-log scale, most vertices (99.91%) are in the **largest** component.  

Think about the way that burner/ghost accounts are created and then applied.  Those are within very small (if any) networks which we are not necessarily concerned about.  

### Degree Distribution of the Internet


