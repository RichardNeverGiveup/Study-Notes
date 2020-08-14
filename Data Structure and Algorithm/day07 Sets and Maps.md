
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
Here, we use __Linear Probing__ to solve the collision problem. When a collision occurs while using linear probing, we advance to the next location in the list to see if that location might be available.


```python

```
