
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
paths in the graph have directions, like {0,1} is not equal to {1,0}.
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
1. The next shortest edge is examined. If the two vertex end points of the edge are in different sets, the edge can be safely added to the set of spanning tree edges. A new set is formed from the union of the two vertex sets and the two previous sets are dropped from the list of vertex sets.
2. If the two vertex endpoints of the edge are already in the same set, the edge is ignored.  



```python

```


```python

```
