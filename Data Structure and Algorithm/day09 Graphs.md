
Abstracting away the details of a problem and studying it in its simplest form often leads to new insight.  
## 0. Graph Notation
#### Cardinality
The number of elements in a set, can be written as |V|. A graphG = (V,E) is defined by a set of __vertices__, named V, and a set of __edges__, named E. Edges are denoted by pairs of vertices.  
eg:  
The sets V = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} and E = {{0, 1},{0, 3},{0, 10},{1, 10},{1, 4},{2, 3},{2, 8},{2, 6},{3, 9},{5, 4},{5, 12},{5, 7},{11, 12},{11, 10},{9, 10}}  
#### Path
a series of edges, none repeated, that can be traversed in order to travel from one vertex to another in a graph. 
#### Cycle 
a path which begins and ends with the same vertex.  
#### Directed Graph
paths in the graph have directions, E is a set of tuples instead of subsets. like {(0,1),(2,3)}
#### Weighted Graph
a graph where edge has a weight assigned to it.  

_Eg_ : how many colors it would take to color a map so that no two countries that shared aborder were colored the same ?  
This problem can be restated as finding the minimum number of colors required to color each vertex in the graph so that no two vertices that share an edge have the same color.   
## 1. Searching a Graph
How can we discover a path from one vertex to another in a graph ? Does a path exist from Vi to Vj ? If it exists, what edges must be traversed to get there?  Maybe we can use depth first search, but we must be wary of __getting stuck in a cycle within a graph__.  
During the search process, if we encountered a vertex with more than one edge, what should we do ? If we choose one edge among the choices, it might lead to a wrong way. So we need a way of backing up and trying the other edges. This is __depth first search__, and the ability to back up to fix that choice (bakctracking) is implemented by a __stack data structure or recursion__ .  
Depth first search must avoid possible cycles within the graph, this is accomplished by maintaining a __set of visited vertices__.  
An iterative (i.e. non-recursive) graph depth first search algorithm begins by initializing the visited set to the empty set and by creating a stack for backtracking.


```python
def graphDFS(G, start, goal):
    # G = (V, E) is the graph with vertices, V, and edges, E.
    V, E = G
    # use this stack to store vertices and backtracking
    stack = Stack()
    # use this set to avoid cycle
    visited = Set()
    stack.push(start)
    
    while not stack.isEmpty():
        current = stack.pop()
        visited.add(current)
        
        if current == goal:
            return True
        
        for v in adjacent(current, E):
            # for every adjacent vertex, v, to the current vertex, v is pushed on the stack of vertices
            # unless v is already in the visited set.
            if not v in visited:
                stack.push(v)
    return False        
```

It can also be implemented recursively if __pushing on the stack__ is replaced with a __recursive call to depth first search__. When implemented recursively, the __depth first search function__ is passed the __current vertex__ and a __mutable visited set__ and it returns either the path to the goal or alternatively a boolean value indicating that the goal or target was found.  

When we want to change a program __from recursive to iterative__, maybe a __stack__ could help!  

The iterative version of __depth first search__ can be modified to do a __breadth first search__ of a graph if the __stack__ is replaced with a __queue__. Breadth first search is an exhaustive search, meaning that it looks at all paths at the same time. Using breadth first search on large graphs may take too long.

## 2. Kruskal's algorithm
A tree is just a graph without any cycles. And a tree must contain one less edge than its number of vertices. There may be many cycles within the graph, to find a __minimum weighted spanning tree__ for such a graph, we could use Kruskal's algorithm.  
It is a greedy algorithm. Greedy means that the algorithm always chooses the first alternative when presented with a list of alternatives. In other words, __no backtracking__ is required in Kruskal's algorithm.  

First we need to __sort all the edges__ in ascending order of their weights. If the graph is fully connected, the spanning tree will contain |V|-1 edges. Then form __sets of all the vertices in the graph, one set for each vertex__. After that we will do the following until |V|-1 edges have been added to the set of spanning tree edges:
1. The shortest edge is examined. If the two vertex end points of the edge are in different sets, the edge can be safely added to the set of spanning tree edges. A new set is formed from the union of the two vertex sets and the two previous sets are dropped from the list of vertex sets.
2. If the two vertex endpoints of the edge are already in the same set, the edge is ignored.  


### Complexity Analysis
Sorting the list of edges is O(|E|log|E|), forming the union of sets takes three steps:
1. discover the set for each endpoint of the edge being considered
2. compare the equality of the two sets
3. union be formed and update the sets of the original vertex to the new union set.

we could use a __list of sets__ to implement, each position in the list corresponded to one vertex in the graph.  
initially the list is `[{0}, {1}, {2}, {3}, {4}]`, then it would be `[{0,1}, {0,1}, {2}, {3}, {4}]`, if the first edge is (0,1)  
The set corresponding to a vertex can be determined in O(1) time since indexed lookup in a list is a constant time operation.  
If we make sure there is only one copy of each set, we can determine if two sets are the same or not in O(1) time as well.We can just compare their references to see whether they are the same set or not. The following code shows this logic.  
`origin = [{0},{1},{2},{3},{4}]
origin[0] = origin[0].union(origin[1])
origin[1] = origin[0]
origin[0] is origin[1]`  
The third operation requires forming a new set from the previous two. This operation will be performed |V|-1 times. In the worst case the first time this operation occurs 1 vertex will be added to an existing set. The second time, two vertices will be added to an existing set, and so on. The overall worst case complexity of this operation is O(|V|^2). So we may need a new data structure to improve this.  
### The Partition Data Structure
The partition data structure contains a list of integers with one entry for each vertex. Initially, the list simply contains a list of integers which match their indices:  
<li> $~$ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29</li>
<li>[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]</li>

Think of this trees as a list of trees, representing the sets of connected edges in the spanning forest constructed so far. __A tree's root is indicated when the value at a location within the list matches its index.__  __Discovering the set for a vertex means tracing a tree back to its root.__   
consider what happens when the edge from vertex 3 to vertex 9 is considered for adding ? The partition at that time looks like this:
<li> $~$ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29</li>
<li>[ 4,4,2,7,5,11,2,7,2,11,11,7,11,16,16,16,16,19,19,19,19,22,22,24,26,26,19,26,29,29]</li>

Vertex 3 is not the root of its own tree at this time. Since 7 is found at index 3, we next look at index 7 of the partition list. That position in the partition list matches its value. The 3 is in the set (i.e. tree) rooted at location 7.    
Let's see vertex 9, index 9 in the list contains 11. Index 11 in the list contains 7. Vertex 9 is also in the set (i.e. tree) rooted at index 7.  
Therefore vertex 3 and 9 are already in the same set and the edge from 3 to 9 cannot be added.

If we consider adding edge(2,3), we will find they are in different sets because their root are different. Then we need to merge these two sets, how ? __By make the root of one of the trees point to the root of the other tree__.  
Then we get this partition:
<li> $~~$ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29</li>
<li>[ 4,4,2,7,5,11,2,2 2,11,11,7,11,16,16,16,16,19,19,19,19,22,22,24,26,26,19,26,29,29]</li>

The partition data structure combines the __three required operations__ into one method called __`sameSetAndUnion`__. This method __receives two vertex numbers and returns Ture if the vertices are in a same set(have same root). If they do not have the same root, the root of one tree is made to point to the other and return false__.  
The sameSetAndUnion method first finds the roots of the two vertices given to it. In the worst case this could take O(|V|) time leading to an overall complexity of O(|V|^2), because there are |V-1| edges. However, in practice these set trees are very flat.  
The average case complexity of sameSetAndUnion is much closer to O(log|V|). This means that the second part of Kruskal's algorithm, using this partition data structure, exhibits O(|E|log|V|) complexity in the average case.  
So the overall average complexity of K's algorithm is O(|E|log|E|).




```python

```


```python

```


```python

```


```python

```
