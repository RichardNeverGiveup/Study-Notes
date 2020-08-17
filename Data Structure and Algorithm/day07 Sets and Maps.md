
## 0. Warming Up
Sudoku puzzle: finde the correct numbers to fill a (9,9) matrix. Each row must have one each of 1-9 in it. The same is true for each column. Finally, there are nine (3,3) squares within the (9,9) matrix that must also have each of 1-9 in them.  

A group is a collection of nine cells that appear in a row, column, or square within the puzzle. Within each cell is a set of numbers representing the possible values for the cell at some point in the process of reducing the puzzle.  
## 1. Sets
Cardinality of a set is the number of items in it(remember elements in set are unique).
### Hashing
Accessing any location within a list can be accomplished in O(1) time. Randomly accessible list means any location within the list can be accessed in O(1) time. But we need the index of the location we wish to access. The index serves as the address of an item in the list.  
If we wanted to implement a set that could test membership in O(1) time, we need to remember the index of each item. How? we need build a relation between the index of the item and the item, __so by proving the item(not index of that item), we could calculate the index(the address of that item).__   

Since each object is stored as 1s and 0s, these 1s&0s can be interpreted as the index. Python provides hash function called on any object to return an integer value, this value is called _hash code_ or _hash value_.   
Mutable objects like lists may not be hashable because when an object is mutated its hash value may also change.

## 2. HashSet Class
We can use a hash value(generated from the item) to compute an index into a list to obtain O(1) item lookup complexity. Now we implement our own HashSet Class.


```python
class HashSet:
    def __init__(self, contents=[]):
        self.items = [None]*10
        self.numItems = 0
        
        for item in contents:
            self.add(item)
```

Since the list must be shorter than the maximum hash value, so we divide hash values by the length of the list. The remainder will always be between 0 and the length of the list minus one.  
Hash values are not necessarily unique. Hash values are integers and there are only finitely many integers possible in a computer. Besides, the remainders will be even less unique than the original hash values.  

__Collision__: When two objects need to be stored at the same index within the hash set list, because their computed indices are identical.   
Here, we use __Linear Probing__ to solve the collision problem. When a collision occurs while using linear probing, we advance to the next location in the list to see if that location might be available. A `None` or a `__Placeholder` object indicates an available location within the hash set list.


```python
def __add(item, items):
    idx = hash(item) % len(items)
    loc = -1
    
    while items[idx] != None:
        if items[idx] == item:
            return False
        
        if loc < 0 and type(items[idx]) == HashSet.__Placeholder:
            loc = idx
        idx = (idx + 1) % len(items)
        
    if loc < 0:
        loc = idx
    items[loc] = item
    
    return True
```

The while loop in the previous code is the _linear probing_ part. The index _idx_ is incremented and probing all the list until item or None is found. Finding a None indicates the end of the linear chain and we must know if the item is already in the set before the end of linear searching. If the item is not in the list, then the item is added either at the location of the first `__Placeholder`
object found in the search, or at the location of the `None` value at the end of the chain.

There is a issue we must address now. If there is __only one position was open__ in the hash set list or If the __list were full__, what would happen? The linear search would result in searching the __entire list__ or the result would be an __infinite loop__.  
We want to add an item in amortized O(1) time. To insure that we get an amortized complexity of O(1), the list must never be full or almost full.  
#### Load Factor
The fullness of the hash set list is called its load factor. By dividing the number of items stored in the list by its length, we can get the load factor of a hash set. A really small load factor means the chance of collision is small. A high load factor means more efficient space utilization, but higher chance of a collision. A reasonable maximum load factor is 75% full.  
#### Rehashing
When adding a value into the list, if the resulting load factor is greater than 75% then all the values in the list must be transferred to a new list. To transfer the values to a new list the values must be hashed again because the new list is a different length. This process is called rehashing.


```python
def __rehash(oldList, newList):
    for x in oldList:
        if x != None and type(x) != HashSet.__Placeholder:
            HashSet.__add(x, newList)
    
    return newList

def add(self, item):
    if HashSet.__add(item, self,items):
        self.numItems += 1
        load = self.numItems / len(self.items)
        if load >= 0.75:
            self.items = HashSet.__rehash(self.items, [None]*2*len(self.items))
```

Deleting a value from a hash set means first finding the item. This may involve doing a linear search in the chain of values. If the value to be deleted is the last in a chain then it can be replaced with a None. If it is in the middle of a chain then we cannot replace it with None because this would cut the chain of values. Instead, the item is replaced with a `__Placeholder` object. A place holder object does not break a chain and a linear probe continues to search skipping over placeholder objects when necessary. 


When removing an item, the load factor may get too low to be efficiently using space in memory. When the load factor dips below 25%, the list is again rehashed to decrease the list size.


```python
class __Placeholder:
    def __init__(self):
        pass
    
    def __eq__(self, other):
        return False

def __remove(item, items):
    idx = hash(item) % len(items)
    
    while items[idx] != None:
        if items[idx] == item:
            nextIdx = (idx + 1) % len(items)
            if items[nextIdx] == None:
                items[idx] = None
            else:
                items[idx] = HashSet.__Placeholder()
            return True
        
        idx = (idx + 1) % len(items)
        
    return False

def remove(self, item):
    if HashSet.__remove(item, self.items):
        self.numItems -= 1
        load = max(self.numItems, 10) / len(self.items)
        if load <= 0.25:
            self.items = HashSet.__rehash(self.items, [None]*int(len(self.items)/2))
    else:
        raise KeyError("Item not in HashSet")
```

Finding an item results in O(1) amortized complexity as well. The chains are kept short as long as most hash values are evenly distributed and the load factor is kept from approaching 1.  

The discard method is nearly the same of the remove method, except that no exception is raised if the item is not in the set when it is discarded.


```python
def __contains__(self, item):
    idx = hash(item) % len(self.items)
    while self.items[idx] != None:
        if self.items[idx] == item:
            return True
        
        idx = (idx + 1) % len(self.items)
    return False
```


```python
def __iter__(self):
    for i in range(len(self.items)):
        if self.items[i] != None and type(self.items[i]) != HashSet.__Placeholder:
            yield self.items[i]
```


```python
def difference_update(self, other):
    for item in other:
        self.discard(item)
        
def difference(self, other):
    result = HashSet(self)
    result.difference_update(other)
    return result
```

## 3.Maps
A map, or dictionary, is a lot like a set. A set and a dictionary both contain uniquevalues.   
The set datatype contains a group of __unique values__.   
A map contains a set of __unique keys__ that map to associated values.  
The interesting difference is that key/value pairs are stored in the dictionary as opposed to just the items of a set. The key part of the key/value pair is used to determine if a key is in the dictionary as you might expect.  

A private `__KVPair` class is defined. Instances of `__KVPair` hold the key/value pairs as they are added to the HashMap object.  
With the addition of a `__getitem__` method on the _HashSet_ class, the _HashSet_ class could be used for the _HashMap_ class implementation.


```python
def __getitem__(self, item):
    idx = hash(item) % len(self.items)
    while self.items[idx] != None:
        if self.items[idx] == item:
            return self.items[idx]
        idx = (idx + 1) % len(self.items)
    return None
```


```python
class HashMap:
    class __KVPair:
        def __init__(self, key, value):
            self.key = key
            self.value =value
            
        def __eq__(self, other):
            if type(self) != type(other):
                return False
            return self.key == other.key
        
        def getKey(self):
            return self.key
        
        def getValue(self):
            return self.value
        
        def __hash__(self):
            return hash(self.key)
    
    def __init__(self):
        self.hSet = hashset.HashSet()
        
    def __len__(self):
        return len(self.hSet)
    
    def __contains__(self, item):
        return HashSet.__KVPair(item, None) in self.hSet
    
    def not__contains__(self, item):
        return item not in self.hSet
    
    def __setitem__(self, key, value):
        self.hSet.add(HashMap.__KVPair(key, value))
        
    def __getitem__(self, key):
        if HashMap.__KVPair(key, None) in self.hSet:
            val = self.hSet[HashMap.__KVPair(key, None)].getValue()
            return val
        
        raise KeyError("Key" + str(key) + " not in HashMap")
        
    def __iter__(self):
        for x in self.hSet:
            yield x.getKey()
```

### Memorization
The idea behind memoization is to do the work of computing a value in a function once. Then, we make a note to ourselves so when the function is called with the same arguments again.


```python
def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n-1) + fib(n-2)
```

fib(5) would call fib(4) and fib(3), fib(4) would call fib(3) and fib(2), fib(3) would call fib(2) and fib(1), by this pattern, if we want to computer fib(5), we would computer fib(3) twice and fib(2) three times. This is called exponential growth.
The complexity of the fib function is O(2^n).


```python
memo = {}

def fib(n):
    if n in memo:
        return memo[n]
    
    if n == 0:
        memo[0] = 0
        return 0
    
    if n == 1:
        memo[1] = 1
        return 1
    
    val = fib(n-1) + fib(n-2)
    
    memo[n] = val
    
    return val
```
