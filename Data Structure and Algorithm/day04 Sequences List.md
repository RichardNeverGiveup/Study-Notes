
## 0. Warming Up
Computers are good at doing something over and over again. When we want do the same thing to similar data or objects, we want to organize those objects into some kind of structure so that program can easily switch from one object to the next.  
An __Abstract Data Type__ is a term that is used to describe a way of organizing data.

## 1. Lists
The PyList we defined earlier uses the built-in list only for setting and getting elements in a list.  
And RAM makes the __indexed get__ and __indexed set__ operations have O(1) complexity.  


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
```

### Constructor
We'll keep track of the number of locations being used and the actual size of the internal list in our PyList objects.
1. the list itself called items
2. the size of the internal list called size
3. the number of locations in the internal list that are currently being used called numItems
  
We wouldn't have to keep track of the size of the list because we can use _len_ funciton. But we still did that,because we don't want to have a lot of _len_ to messy the code. 

All the used locations in the internal list will occur at the beginning of the list. __Why don't we just put holes among the internal list?__   
Storing all the items at the beginning of the list, without holes, also means that we can randomly access elements of the list in O(1) time. We do not have to search for the proper location of an element. Indexing into the PyList will simply index into the internal items list to find the proper element as seen in the next sections.


```python
class PyList:
    def __init__(self, contents=[], size=10):
        self.items = [None]*size
        self.numItems = 0
        self.size = size
        for e in contents:
            self.append(e)
            
    def append(self, item):
        if self.numItems == len(self.items):
            newlst = [None]*self.numItems*2
            for k in range(len(self.item)):
                newlst[k] = self.items[k]
            self.items = newlst
        self.items[self.numItems] = item
        self.numItems += 1
                        
test = PyList(['a','b','c'])
'''at this time, we did not impletement __str__() method __repr__() or __iter__() method in this class, so we
can not show the elements in the list by print or use a for loop. Without __getitem__() method, indexing the element
or slice the elements.
'''
```

The complexity of creating a PyList object is O(1) if no value is passed to the constructor and O(n) if a sequence is passed to the constructor.

### Get and Set
Our PyList class is a wrapper for the built-in list class. So, to implement the get item and set item operations on PyList, we will use the get and set operations on the built-in list class. The code is given here. The complexity of both operations is O(1).


```python
def __getitem__(self, index):
    if index >=0 and index < self.numItems:
        return self.items[index]
    raise IndexError("PyList index out of range")
    
def __setitem__(self, index, val):
    if index >= 0 and index < self.numItems:
        self.items[index] = val
        return
    raise IndexError("PyList assignment index out of range")
```

### Concatenate and Append
To concatenate two lists we must build a new list that contains the contents of both. This is an accessor method because it does not mutate either list. Instead, it builds a new list.The complexity is O(n) where n is the sum of the lengths of the two lists.  


```python
def __add__(self, other):
    result = PyList(size=self.numItems + other.numItems)
    
    for i in range(self.numItems):
        result.append(self.items[i])
    for i in range(other.numItems):
        result.append(other.items[i])
    return result
```

It turns out that adding 25% more space each time is enough to guarantee O(1) complexity. Even if we added even 10% more space each time we would get O(1) complexity.  
Integer division by 4 is very quick in a computer because it can be implemented by shifting the bits of the integer to the right.


```python
class PyList:
    def __init__(self, contents=[], size=10):
        self.items = [None]*size
        self.numItems = 0
        self.size = size
        for e in contents:
            self.append(e)
    
    def __makeroom(self):
        # add one in case for some reason self.size is 0.
        newlen = (self.size // 4) + self.size + 1
        newlst = [None] * newlen
        for i in range(self.numItems):
            newlst[i] = self.items[i]
        self.items = newlst
        self.size = newlen
        
    def append(self, item):
        if self.numItems == self.size:
            self.__makeroom()
            
        self.items[self.numItems] = item
        self.numItems += 1
```

### Insert
If size is full, make some room first, because the elements after the insertion point need to move to the right.

This works best if we start from the right end of the list and work our way back to the point where the new value will be inserted.
If not backward, the element in the insertion point i would move to the right, like `li[i+1] = li[i]`. But when we want to move the
next element i+1 to i+2, the original i+1 is lost.  

Besides, we need to accumulate the numItems by 1 after moving is done. Otherwise, if we increase it before the moving is done, the indice of the original list got messy.

The complexity of this operation is O(n) where n is the number of elements in the list after the insertion point.  
If the index provided is larger than the size of the list the new item, e, is appended to the end of the list.


```python
def insert(self, i, e):
    if self.numItems == self.size:
        self.__makeroom()
    if i < self.numItems:
        for j in range(self.numItems - 1, i-1, -1):
            self.items[j + 1] = self.items[j]
            
        self.items[i] = e
        self.numItems += 1
    else:
        self.append(e)
```

### Delete
The complexity of delete is O(n) in the average.  
To conserve space, in python, if a list reaches a point after deletion where less than half of the locations within the internal list are being used, then the size of the available space is reduced by one half.  
`def __delitem__(self,index):
    for i in range(index, self.numItems-1):
        self.items[i] = self.items[i+1]
        self.numItems -= 1`


```python
def __eq__(self,other):
    if type(other) != type(self):
        return False

    if self.numItems != other.numItems:
        return False

    for i in range(self.numItems):
        if self.items[i] != other.items[i]:
            return False

    return True
```

The yield call in Python suspends the execution of the `__iter__` method and returns the yielded item to the iterator.    
`def __iter__(self):
    for i in range(self.numItems):
        yield self.items[i]`  
This iter function takes advantage of built-in list. Since self.numItems is a built-in list, then it can be looped.


```python
def __len__(self):
    return self.numItems

def __contains__(self,item):
    for i in range(self.numItems):
        if self.items[i] == item:
            return True

    return False

def __str__(self):
    s = "["
    for i in range(self.numItems):
        s = s + repr(self.items[i])
        if i < self.numItems - 1: # from 0 to numItems-1, this line makes the elements form 0 to numItems-2 are seperated with comma
            s = s + ", "
    s = s + "]"
    return s

def __repr__(self):
    s = "PyList(["
    for i in range(self.numItems):
        s = s + repr(self.items[i])
        if i < self.numItems - 1:
            s = s + ", "
    s = s + "])"
    return s
```

Python includes a function called eval that will take a string containing an expression and evaluate the expression in the string. For instance, `eval("6+5")` results in 11 and `eval("[1,2,3]"`) results in the `list [1,2,3]`. The `__repr__` method, if defined, should return a string representation of an object that is suitable to be given to the `eval` function.

### Cloning Objects
`eval(repr(x))` is a copy of object x. Since all the tiems in the PyList object x are cloned by evaluating the representation of the object. Cloning an object like this is called __deep copy__.  
__shallow copy__: When an object is cloned, but items in the original object are shared with the clone.  
`x = PyList([1,2,3])
 y = PyList(x)`  
 This shallow copy is ok, because integers 1,2,3 are immutable items. But we should be cautious when dealing with shallow copy of mutable items.
