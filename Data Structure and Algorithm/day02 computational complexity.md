
## 1. The behavior of RAM
1. RAM of a computer does not behave like a post office.  
The computer does not need to find the right RAM location before the it can retrieve or store a value.  
2. RAM is like a group of people, each person is assigned an address.  
It does not take any time to find the right person because all the people are listening, just in case their name is called.  

All locations within the RAM of a computercan be accessed in the same amount of time.  
So this characteristic leads two results in python and we can test them:  

__a__. The size of a list does not affect the average access time in the list.  
__b__. The average access time at any location within a list is the same, regardless of its location within the list.


```python
import datetime
import random
import time

# the size of test lists range from 1000 to 200000
xmin = 1000  # sizemin
xmax = 200000  # sizemax

# record the list sizes in sizeList
# record average access time of a list for 1000 retrievals in timeList
xList = []  # sizeList
yList = []  # timeList

for x in range(xmin, xmax+1,1000):
    xList.append(x)
    prod = 0
    # creat a list of 0s, size is x (1000,2000,....)
    lst = [0]*x
    
    starttime = datetime.datetime.now()

    for v in range(1000):
        index = random.randint(0,x-1)
        val = lst[index]
        # dummy operation
        prod = prod*val
    endtime = datetime.datetime.now()

    delatT = endtime - starttime
    # Divide by 1000 for the average access time, but also multiply by 1000000 for microseconds
    accessTime = delatT.total_seconds()*1000
    yList.append(accessTime)

spec = max(yList) - min(yList)  # this is the variation in microseconds
```




    1.0749999999999997



## 2. Big-Oh Notation
The actual time to access the RAM and finish a program may vary, but at least there is an upper bound of how
much time it will take. The idea of an upper bound is called Big-Oh notation.

The size of the list does not change the time needed
to access or store an element and there is a fixed upper bound for the amount of time
needed to access or store a value in memory or in a list.


```python
class PyList:
    def __init__(self):
        self.items = []

    def append(self,item):
        self.items = self.items + [item]
```

#### This PyList append function did three steps:
1. The item itself is not changed. A new list is constructed from the item.
2. The + opertator is an accessor method that does not change either original list.
3. The assignment of self.items to this new list updates the PyList object so it now refers to the new list.

So if we want to know the time it takes to append n elements to the PyList, we need to know:  
<strong>all the list accesses * (time to access a list element + time to store a list element)</strong>
    
#### Q. How much time it takes to append n items to a PyList?
First time we called append we had to copy one element of the list.The second time we copy two element....  
1 + 2 + 3 + 4 + ... + n = n*(n+1)/2. So it's O(n2)  
#### Q. How to improve?
It more efficient if we didn鈥檛 have to access each element of the first list when
concatenating the two lists. The use of the + operator is what causes Python to
access each element of that first list. When + is used a new list is created with space
for one more element. Then all the elements from the old list must be copied to the
new list and the new element is added at the end of this list.

## 3. Commonly Occurring Computational Complexities
O(1) < O(log n) < O(n log n) < O(n^2) < (c^n).  
Function T(n) that is a description of the running time of an algorithm, where n is the size of the data given to the algorithm.  
We want to study the asymptotic behavior = we want to study how T(n) increases as n鈫?鈭?.  

Omega notation $\Omega(g(n))$ serves as a way to describe a lower bound of a function.  
With in the lower and upper bound, in some cases we can get the tight bound Theta notation $\Theta (g(n))$   

## 4. Amortized Complexity
Sometimes it is not possible to find a tight upper bound on an algorithm.  
For instance, most operations may be bounded by some function c*g(n), but every once in a while there may be an operation that takes longer.  
#### Amortized Complexity
A method to get as tight an upper bound as we can for the worst case running time of any sequence of n operations on a data structure.  

Python is implemented in C and python lists are implemented as arrays in C, but python lists support effective append method.  
In C, an array can be allocated with a fixed size, but cannot have its size increased once created. Writing [None]*n creates a fixed size list of n elements each referencing the value None.
#### So how can python support append?
When python list runs out of space in the fixed size list, will double the size of the list copying all items from the old list to the new list .


```python
class PyList:
    def __init__(self, size=1):
        self.items = [None] * size
        self.numItems = 0
        
    def append(self, item):
        if self.numItems == len(self.items):
            newlst = [None] * self.numItems * 2
            for k in range(len(self.items)):
                newlst[k] = self.items[k]
            self.items = newlst
        self.items[self.numItems] = item
        self.numItems += 1
def main():
    p = PyList()

    for k in range(100):
        p.append(k)
        print(p.items)
        print(p.numItems)
        print(len(p.items))
        
if __name__ == "__main__":
    main()
```
Using this new Pylist append method, a sequence of n append operations takes O(n) time. This means individual operations must not take longer than O(1) time. But when the list run out of space a new list is created and all the old elements are copied to the new list, copying n elements takes longer than O(1) time. Technically, when the list size is doubled the complexity of append is O(n)  
However, amortized complexity of append is O(1) since the cases the list needs to be doubled in not so often.
