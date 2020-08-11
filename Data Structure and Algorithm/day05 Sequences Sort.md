
## 0.Warming Up
We can define the less than operator (i.e. <) by writing an `__lt__` method in the class. Once defined, this less than operator orders all the elements of the class.  
Strings are compared lexicographically, this means sorting a sequence of strings will end up alphabetized like you would see in a dictionary.  
For lists to have an ordering, the elements at corresponding indices within the lists must be orderable.  
`lst = [1,2,3]
 lst2 = list("abc")
 lst < lst2`  
Comparing lst and lst2 did not work because the items in the two lists are not orderable. We can't compare an integer and a string.  
But we can compare `[1,1,"a"]` with lst. List comparison is performed lexicographically like strings, so under this circumstance, 3 and "a" are not required to be compared.


```python
import turtle

class Point(turtle.RawTurtle):
    def __init__(self, canvas, x, y):
        super().__init__(canvas)
    
    def __lt__(self, other):
        return self.ycor() < other.ycor()
```

## 1. Seleciton Sort
This algorithm begins by finding the smallest value to place in the first position in the list. It does this by doing a linear search through the list and along the way remembering the index of the smallest item it finds.  
First guessing that the smallest item is the first item in the list and then checking the subsequent items, if a small one on position j is found, then we guess the element in position j is the smallest and keep searching forward. 


```python
def select(seq, start):
    minIndex = start
    for j in range(start+1, len(seq)):
        if seq[minIndex] > seq[j]:
            minIndex = j
    return minIndex

def selSort(seq):
    for i in range(len(seq)-1):
        minIndex = select(seq, i)
        tmp = seq[i]
        seq[i] = seq[minIndex]
        seq[minIndex] = tmp
```

## 2. Merge Sort
### Divide and conquer algorithms: 
usually written recursively, but don鈥檛 necessarily have to be. The basic premise is that we divide a problem into two pieces. Each of the two pieces is easier to solve.  
### Merge sort:
divides the list again and again, until we are left with lists of size 1. Two sorted sublists can be merged into one sorted list in O(n) time. A list can be divided into lists of size 1 by repeatedly splitting in O(log n) time.  

The merge function takes care of merging two adjacent sublists. The first sublist runs from __start__ to __mid-1__. The second sublist runs from __mid__ to __stop-1__. The elements of the two sorted sublists are copied, in O(n) time, to a new list. Then the sorted list is copied back into the original sequence, again in O(n) time.


```python
def mergeSortRecursively(seq, start, stop):
    # stop-1 is the index of last element. when [1], start=0, stop=1. so base case is start == stop-1
    # when [], empty list, start=0, stop=0, so we need start>stop-1 to cover this condition.
    # may be using start >= stop-1 is hard to understand. just remember we split the list into two half recursively
    # if start < stop-1, so the following code is the same.
#     if start < stop-1:
#         mergeSortRecursively(seq, start, mid)
#         mergeSortRecursively(seq, start, stop)
#         merge(seq, start, mid, stop)
#     return
    
    if start >= stop-1:
        return
    mid = (start+stop)//2 # say we have 4 elements, start is always 0, so stop-1 is index of the last el(3)
    # mid = (0+4)//2 = 2. [0,1,2,3] -> [start,mid-1]&[mid,stop-1] -> [0,1]&[2,3]
    mergeSortRecursively(seq, start, mid)
    mergeSortRecursively(seq, mid, stop)
    merge(seq, start, mid, stop)
    
def merge(seq, start, mid, stop):
    lst = []
    i = start
    j = mid
    while i < mid and j< stop:
        if seq[i] < seq[j]:
            lst.append(seq[i])
            i += 1
        else:
            lst.append(seq[j])
            j += 1
    while i < mid:
        lst.append(seq[i])
        i += 1
        
    for i in range(len(lst)):
        seq[start + i] = lst[i]
        
def mergerSort(seq):
    mergeSortRecursively(seq, 0, len(seq))
```


```python
seq = [7,5,2,8,1,3,4,6]
mergerSort(seq)
seq
```




    [1, 2, 3, 4, 5, 6, 7, 8]




