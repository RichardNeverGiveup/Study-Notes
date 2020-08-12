
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
usually written recursively, but dont necessarily have to be. The basic premise is that we divide a problem into two pieces. Each of the two pieces is easier to solve.  
### Merge sort:
divides the list again and again, until we are left with lists of size 1. Two sorted sublists can be merged into one sorted list in O(n) time. A list can be divided into lists of size 1 by repeatedly splitting in O(log n) time.  

The merge function takes care of merging two adjacent sublists. The first sublist runs from __start__ to __mid-1__. The second sublist runs from __mid__ to __stop-1__. The elements of the two sorted sublists are copied, in O(n) time, to a new list. Then the sorted list is copied back into the original sequence, again in O(n) time.

One criticism of the merge sort algorithm is that the elements of the two sublists cannot be merged without copying to a new list and then back again. Other sorting methods, like Quicksort, have the same O(n log n) complexity as merge sort and do not require an extra list.  
The repetitive splitting of the list results in O(log n) splits. Since there are log n levels to the merge sort algorithm and each level takes O(n) to merge, the algorithm is O(n log n).


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
    
# the difficult part is this merger part, use small test cases to draw illustration could help    
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
#     we do not need copy the rest of the sequence from j to stop to lst here, because if we did that, 
#     in the following code the same part of the sequence would be copied right back to the same place.    
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



## Quicksort
Quicksort is also a divide and conquer algorithm and is usually written recursively. But, where merge sort splits the list until we reach a size of 1 and then merges sorted lists, the quicksort algorithm does the merging first and then splits the list. We cant merge an unsorted list. So we partition it into two lists. What we want is to prepare the list so quicksort can be called recursively. But, if we are to be successful, the two sublists must somehow be easier to sort than the original. This preparation for splitting is called partitioning.  
To partition a list we pick a pivot element. Think of quicksort partitioning a list into all the items bigger than the pivot and all the elements smaller than the pivot.We put all the bigger items to the right of the pivot and all the littler items to the left of the pivot.  

After this we can assure:  
1. The pivot is in its final location in the list.  
2. The two sublists are now smaller and can therefore be quicksorted.

But picking the pivot is an art, if we want best performance, we need the pivot in the middle of the sequence.  
For quicksort to have O(n log n) complexity, like merge sort, it must partition the list in O(n) time. If we dont choose a pivot close to the middle, we will not get the O(n log n) complexity we hope for. One way is starting by randomizing the sequence.


```python
import random

def partition(seq, start, stop):
    pivotIndex = start
    pivot = seq[pivotIndex]
    i = start + 1
    j = stop - 1
    
    while i <= j:
#         while i <= j and seq[i] <= pivot:
        while i <= j and not pivot < seq[i]:
            i += 1
#         while i <= j and seq[j] >pivot:
        while i <= j and pivot < seq[j]:
            j -= 1
        if i < j:
            tmp = seq[i]
            seq[i] = seq[j]
            seq[j] = tmp
            i += 1
            j -= 1
        
    seq[pivotIndex] = seq[j]
    seq[j] = pivot
    
    return j

def quicksortRecursively(seq, start, stop):
    if start >= stop-1:
        return
    pivotIndex = partition(seq, start, stop)
    
    quicksortRecursively(seq, start, pivotIndex)
    quicksortRecursively(seq, pivotIndex + 1, stop)


def quicksort(seq):
#     randomize the sequence
    for i in range(len(seq)):
        j = random.randint(0,len(seq)-1)
        tmp = seq[i]
        seq[i] = seq[j]
        seq[j] = tmp
    quicksortRecursively(seq, 0, len(seq))
    



```


```python
li = [7,6,4,5,3,2,1]
quicksort(li)
li
```




    [1, 2, 3, 4, 5, 6, 7]



Every time a value bigger than the pivot is found on the left side and a value smaller than the pivot is found on the right side, the two values are swapped. Once we reach the middle from both sides, the pivot is swapped into place.  

In the partition code, the two commented while loop conditions are probably easier to understand than the uncommented code. However, the uncommented code only uses the less than operator. It only requires that the less than operator be defined between items in the sequence.  
If the sequence given to quicksort is sorted, or almost sorted, in either ascending or descending order, then quicksort will not achieve O(n log n) complexity. In fact, the worst case complexity of the algorithm is O(n2).

The algorithm will simply put one value in place and end up with one big partition of all the rest of the values. If this happened each time a pivot was chosen it would lead to O(n2) complexity. Randomizing the list prior to quicksorting it will help to ensure that this does not happen.
