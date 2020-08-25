
## 0. Warming up
Heap sometimes refer to an area of memory used for dynamic(run-time) memory allocation. But the heap we will demonstrate here is a data structre that is a complete binary tree. Heaps are used in implementing priority queues, the heapsort, and some graph algorithms.  
__A largest-on-top heap__ is a complete ordered tree such that every node is >= all of its children.  
__A heap__ is a tree that is full on all levels except possibly the lowest level which is filled in from left to right. But heaps are not stored as trees, because they are complete trees, they may be stored in an arry.  

The children of any element of the array can be calculated from the index of the parent.  
__leftChildIndex = 2 * parentIndex + 1  
rightChildIndex = 2 * parentIndex + 2__  

Given a child's index, we can discover where the parent is located.  
__parentIndex = (childIndex - 1)//2__  
eg: parentIndex = (8 - 1)//2 = 3  
Heaps can be built either largest on top or smallest on top. We call the count of items currently stored in the heap the __size__ of the heap.


```python
def buildFrom(self, aSequence):
    '''aSequence is an instance of a sequence collection which
    understands the comparison operators. The elements of
    aSequence are copied into the heap and ordered to build
    a heap.'''
def __siftUpFrom(self, childIndex):
    '''ildIndex is the index of a node in the heap. This method sifts
    that node up as far as necessary to ensure that the path to the root
    satisfies the heap condition.'''
```

The sequence of values passed to the buildFrom method will be copied into the heap. Then, each __subsequent__ value in the list will be sifted up into its final location in the heap.  
Consider the list of values [71, 15, 36, 57, 101]. To begin, siftUpFrom is called on the second element, the 57 in this case. Normally, the parent index of the node is computed. If the value at the current child index is greater than the value at the parent index, then the two are swapped and the process repeats:  
__until the root node is reached or until the new node is in the proper location to maintain the heap property.__  



```python
#   [71,15,36,57,101]             71                              71                   71                   101
#                                /  \        swap                /  \                 /  \                 /  \
#                              15    36    >>>>>>>>>>          57   36  >>>>>>>>>   101  36  >>>>>>>>    71   36
#                             / \                             / \                  /  \                /  \
#                         (57)   101                        15  (101)            15   57             15   57
#
```

## 1. Heapsort Version1
Heaps have two basic operations. 
1. You can add a value to a heap. 
2. You can also delete, and retrieve, the maximum value from a heap if the heap is a largest on top heap.  
Using these two operations, we can devise a sorting algorithm by building a heap with a list of values and then removing the values one by one in descending order.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
