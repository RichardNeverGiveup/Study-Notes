
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

Using these two operations, we can devise a sorting algorithm by building a heap with a list of values(__phaseI__) and then removing the values one by one in descending order(__phaseII__).  
We'll call these two parts of the algorithm phase I and phase II. To implement phase I we'll need one new method in our Heap class.

### Phase 1  
`def addToHeap(self, newObject):
'''If the heap is full, double its current capacity. Add the newObject to the heap'''`  
This new method can use the `__siftUpFrom` private method to get the new element to its final destination within the heap. Version 1 of Phase I calls addToHeap n times. This results in O(nlogn):  
1. double the capacity of the heap if necessary
2. data[size] = newObject
3. ` __siftUpFrom(size)`
4. size += 1

` __siftUpFrom` will be called n times. once for each element of the heap, each time it is called, the heap will have grown by 1 element.  

Consider a perfect complete binary tree with h levels:
<table>
<tr><td>Level</td><td>nodes at level#</td></tr>
<tr><td>1</td><td>1</td></tr>
<tr><td>2</td><td>2^(2-1)=2</td></tr>
<tr><td>3</td><td>2^(3-1)=4</td></tr>    
<tr><td>......</td><td>......</td></tr>    
<tr><td>h</td><td>2^(h-1)</td></tr>
</table>  

the nodes in the tree can be computed using geometric sequence formula: $n = 2^h -1$.  So h is equal to log, base 2 of n+1, but not every heap is completely full so there may be some values of n that won't give us an integer for h, then we need to round up using the __ceiling operator__.  
To determine the complexity of phase I, we will use the inequality __logI <= log(I+1) <=logI + 1__, the base is 2 and for any I >= 2  
Since we knew the ceilling of $log(I+1)$ is height of the heap(h), and Phase I appends each value to the end of the list where it is sifted up to its final location. Since __sifting up will go through at most h levels and the heap grows by one each time__,   
The following summation describes an upper bound for Phase I:  $\sum_{2}^{N}c log(I+1)$ , _clog means ceilling after log_.  

Then we have $\sum_{2}^{N}logI$ <= $\sum_{2}^{N}clog(I+1)$ <= $\sum_{2}^{N}(logI+1)$ = $(\sum_{2}^{N}logI)+(N-1)$. Using some calculus, we can get the result of $\sum_{2}^{N}logI$, then we can prove the upper and lower bound are the same. So the work done by inserting N elements into a heap using the `__siftUpFrom` method is $\Theta (NlogN)$. But we can do better using another approach to achieve O(N).
### Phase 2
Now we need to take elements out of the heap, one at a time and put them in a list. We will use the original list that was used for the heap to place the returned values to save space. Take one item from the list and places it where it belongs and the size of the heap is decremented by one, the key operation is the `__siftDownFromTo` method.  
Say we have [101,71,36,15,57], 101 would go to the end of the list, so we swap 57 and 101, then 101 is at its final position within a sorted list. The 57 is not in the correct location, so we call the `__siftDownFromTo` methond to sift 57 down form the 0 position to at most the size-1 location. The `__siftDownFromTo` swaps the 57 with the bigger of the two children, the 71. And because 57 is bigger than 15, it will stop. The Phase 2 will keep doing this until the heap is only of size 1.  
The best case of Phase 2 would require that all values in the heap are identical. In that case the computational complexity would be O(N) since the values would never sift down. So we can limit __how far down the value is sifted to speed up Phase 1__.

## Heapsort Version 2
We will speed up Phase 1 of the heapsort algorithm up to O(N) complexity by limiting how far each newly inserted value must be sifted down. Rather than inserting each element at the top of the heap, we'll build the heap, or heaps, from the bottom up, by
starting at the end of the list rather than the beginning.  
Rather than starting from the first element of the list, we'll start from the other end of the list. __There is no need to start with the last element__. We just pick a node that is a parent of some node in the tree.  
Since the final heap is a binary heap, at least half the nodes should be leaf nodes. so we have the __first parent index: 
parentIndex = (size - 2) // 2__. Note because the list is indexed from 0, so we need to subtract 2 to get the proper parentIndex.  
Say we have `[15,-34,23,46,24,20]`, the first parentIndex is (6-2)//2 = 2, so we will start at index 2 to bulid our heaps from the bottom up. __We will sift it down as far as necessary__.  
childIndex1 = 2 * parentIndex + 1 = 2 * 2 + 1 = 5  
childIndex2 = 2 * parentIndex + 2 = 2 * 2 + 2 = 6  
Since the second one is out of the index of the list, the `__siftDownFromTo` will not consider it. After considering the 20 and the 23 we see that those two nodes do in fact form a heap. We only had to sift the parent down one position at the most.   
Next, we move back one more in the list to index 1. `__siftDownFromTo` start from this node, doing so causes the sift down method to pick the larger of the two children to swap with, forming a heap out of the three values `-34, 46, 24`. Then we get `[15,46,23,-34,24,20]`.  
Finally we move backward in the list one more to index 0, this time we only need to look at the values of the two children because they will already be the largest values in their respective heaps.   
Calling `__siftDownFromTo` on the first element of the list will pick the maximum value from 15, 46, and 23, finally we get `[46,15,23,-34,24,20]` This doesn't form a complete heap yet. We still need to move the 15 down again and `__siftDownFromTo` takes care of moving the 15 to the bottom of the heap `[46,24,23,-34,15,20]`  
### Analysis of V2
For a perfect binary tree of height h, containing(2^h-1) nodes, the sums of the lengths of its maximum comparison paths is (2^h-1-h).
<table>
    <tr><td>Level</td><td>Max path length</td><td>nodes at level#</td></tr>
    <tr><td>1</td><td>h-1</td><td>1</td></tr>
    <tr><td>2</td><td>h-2</td><td>2</td></tr>
    <tr><td>3</td><td>h-3</td><td>4</td></tr>
    <tr><td>...</td><td>...</td><td>...</td></tr>
    <tr><td>h-2</td><td>2</td><td>2^(h-3)</td></tr>
    <tr><td>h-1</td><td>1</td><td>2^(h-2)</td></tr>
    <tr><td>h</td><td>0</td><td>2^(h-1)</td></tr>
</table> 

S = 1*(h-1)+2*(h-2)+2^2*(h-3)+...+2^(h-3)*2+2^(h-2)*1  
S would be an upper bound of the work tobe done. We can solve this by S = 2S - S.  
We can get S = 1+2+2^2+...+2^(h-2)+2^(h-1)-h = 2^h - 1 - h and N = 2^h - 1 nodes, so S = O(N)  
Version 2 of phase I is O(N). Phase II is still O(N log N) so the overall complexity of heap sort is O(Nlog N).  
Quicksort is more efficient than heapsort even though they have the same computational complexity. The built-in sort, which is quicksort implemented in C, runs the fastest, due to being implemented in C.
