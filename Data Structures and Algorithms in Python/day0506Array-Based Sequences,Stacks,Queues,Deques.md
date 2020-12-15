```python
# recap of shallow and deep copy

primes = [2,3,5,7,11,13,17,19]  # python's list is referential array.
temp = primes[3:6]   # this is a shallow copy of primes, if the shared element between primes and temp is immutable objects like int,
# and we change an element in either temp or primes, this won't change cause problem.
temp[2] = 15
print(primes)
print(temp)
```

    [2, 3, 5, 7, 11, 13, 17, 19]
    [7, 11, 15]
    

String are represented using array of characters(not array of references), we call this _compact array_. Python's list will typically use 64-bits for the memory address stored in the array. But the Unicode character stored in a compact array within a string usually requires 2 bytes.  
For python list, it will use four to five times as much memory. Compared with directly store int in the array. Python list need to store the 64-bit memory address per entry and the data corresponding to the memory address.  
We can use `import sys` and `sys.getsizeof` to check the memory usage.  
If we want to use compact array, we can use `array` module and there is a class named `array` in this module. The array module doesn't support making compact arrays of user defined types. check `ctypes` module for this support.


```python
import array
import sys

a = array.array('i',[1,2,3])
b = array.array('i',[1,2,3,4])

print(sys.getsizeof(a))
print(sys.getsizeof(b))
```

    76
    80
    


```python
# demo, prove python list is a dynamic array
import sys
data = []
n = 10
for k in range(n):
    a = len(data)
    b = sys.getsizeof(data)
    print('Length: {0:3d}; Size in bytes: {1:4d}'.format(a,b))
    data.append(None)
```

    Length:   0; Size in bytes:   64
    Length:   1; Size in bytes:   96
    Length:   2; Size in bytes:   96
    Length:   3; Size in bytes:   96
    Length:   4; Size in bytes:   96
    Length:   5; Size in bytes:  128
    Length:   6; Size in bytes:  128
    Length:   7; Size in bytes:  128
    Length:   8; Size in bytes:  128
    Length:   9; Size in bytes:  192
    

We can speculate that in a list instance, it maintains state information akin to:  
`_n`: The number of actual elements currently stored in the list.  
`_capacity`: The maximum number of elements that could be stored in the currently allocated array.  
`_A`: The reference to the currently allocated array(initially None).  

Because a list is a referential structure, the result of `getsizeof` for a list instance only includes the size for its primary structure; it does not include the memory used by the `objects` that are elements of the list.  
Suppor for creating low-level array is provided by a module named `ctypes`.


```python
import ctypes

class DynamicArray:
    """A dynamic array class akin to a simplified Python list."""
    def __init__(self):
        self._n = 0          # count actual elements
        self._capacity = 1   # default array capacity
        self._A = self._make_array(self._capacity)    # low-level array
        
    def __len__(self):
        return self._n
    
    def __getitem__(self, k):
        if not 0 <= k < self._n:
            raise IndexError('Wrong Index')
        return self._A[k]          # retrieve from array
    
    def append(self, obj):
        if self._n == self._capacity:
            self._resize(2 * self._capacity)
        self._A[self._n] = obj
        self._n += 1
        
    def _resize(self, c):
        B = self._make_array(c)
        for k in range(self._n):
            B[k] = self._A[k]
        self._A = B
        self._capacity = c
    
    def _make_array(self, c):
        return (c * ctypes.py_object)()
```

Amortization is a design pattern. We can use a example of cyber-dollar to perform amortized analysis.  
Assume that one cyber-dollar can pay for the execution of each append operation, excluding the time spent for growing the array. Also, let's assume that growing from size k to size 2k require k cyber-dollars to initializing the new array( _to copy the elements in the old list to the new list_ ).  

We shall charge each append operation three cyber-dollars. We overcharge each append operation that doesn't cause an overflow by two cyber-dollars. An overflow occurs when the array S has 2^i elements, at that time, we need to double the size of the array, this will require 2^i cyber-dollars. Luckily, these can be found in cells 2^(i-1) through 2^i - 1.  
That is, we can pay for the executation of n append operations using 3n cyber-dollars. In other words, the amortized time of each append is O(1).

data is a python list  
data.count(value)--------------O(n);  
data.index(value)--------------O(k+1);  
value in data   --------------O(k+1);  
data[j:k] ---------------O(k-j+1);  

data.insert(k,value)-----------O(n-k+1);  
data.pop(k)/ del data[k]-------------O(n-k); (find the data in index k)   
data.remove(value)-----------------O(n);  
data1.extent(data2)/ data1 += data2----------------O(n2);  
data.reverse()----------------O(n);


```python
def insert(self, k, value):
    """Insert value at index k, shifting subsequent values rightward."""
    # we assume 0<=k<=n
    if self._n == self._capacity:
        self._resize(2 * self._capacity)
    for j in range(self._n, k, -1):  # tricky reverse way
        self._A[j] = self._A[j-1]
    self._A[k] = value
    self._n += 1
    
def remove(self, value):
    """Remove first occurrence of value.
    We do not shrink the dynamic array in this version.
    """
    for k in range(self._n):
        if self._A[k] == value:
            for j in range(k, self._n-1):
                self._A[j] = self._A[j+1]
            self._A[self._n - 1] = None   # help garbage collection
            self._n -= 1
            return
        raise ValueError('value not found.')
```

`data.extend(other)` produces the same outcome as append one by one, but the greater efficiency of extend is threefold. Beacuse Python's built-in methods are often implemented natively in compiled language(rather than as interpreted Python code) and the resulting size of the updated list can be calculated in advance(no need to resize the dynamic array multiple times). This is also why list comprehension is more efficient compared with append values to an empty list.


```python
class GameEntry:
    """Represents one entry of a list of high scores."""
    def __init__(self, name, score):
        self._name = name
        self._score = score
        
    def get_name(self):
        return self._name
    
    def get_score(self):
        return self._score
    
    def __str__(self):
        return '({0}, {1})'.format(self._name,self._score)

class Scoreboard:
    """Fixed-length sequence of high scores in nondecreasing order."""
    def __init__(self, capacity=10):
        """Initialize scoreboard with given maximum capacity.
        """
        self._board = [None] * capacity
        self._n = 0
    def __getitem__(self, k):
        """Return entry at index k."""
        return self._board[k]
    
    def __str__(self):
        """Return string representation of the high score list."""
        return '\n'.join(str(self._board[j]) for j in range(self._n))
#**************************************************************************************this function is tricky   
    def add(self, entry):
        """Consider adding entry to high scores."""
        score = entry.get_score()
         
        good = self._n < len(self._board) or score > self._board[-1].get_score()
        if good:
            if self._n < len(self._board):
                self._n += 1
            # shift lower scores rightward to make room for new entry
            j = self._n - 1
            while j > 0 and self._board[j-1].get_score() < score:
                self._board[j] = self._board[j-1]     # shift entry from j-1 to j
                j -= 1                                # decrement j
            self._board[j] = entry                  
```


```python
#**************************************************************************************this function is tricky 
def insertion_sort(A):
    """Sort list of comparable elements into nondecreasing order."""
    for k in range(1, len(A)):
        cur = A[k]       # current element to be inserted
        j = k           
        while j > 0 and A[j-1] > cur:
            A[j] = A[j-1]
            j -= 1
        A[j] = cur
```

The __adapter__ design pattern applies to any context where we effectively want to modify an existing class so that its methods match those of a ralated, but different, class or interface.  
eg: define a new class in such a way that it contains an instance of the existing class as a hidden field.  
Stack Method:  
S.push(e)-----------list.append(e)  
S.pop()-----------list.pop()  
S.top()-----------list[-1]  
S.is_empty()----------len(list)==0  
len(S)------------len(list)


```python
class ArrayStack:
    def __init__(self):
        self._data = []
        
    def __len__(self):
        return len(self._data)
    
    def is_empty(self):
        return len(self._data) == 0
    
    def push(self, e):
        self._data.append(e)
    
    def top(self):
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data[-1]
    
    def pop(self):
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data.pop()
    
# we might wish for the constructor to accept a parameter specifying the maximum capacity of a stack and to initialize the _data 
# member to a list of that length. This means we need a seperate integer as an instance variable to record the current number of 
# elements in the stack.
```


```python
# LIFO can be used as a tool to reverse a data sequence.
def reverse_file(filename):
    """Overwrite given file with its contens line-by-line reversed."""
    S = ArrayStack()
    original = open(filename)
    for line in original:
        S.push(line.rstrip('\n'))
    original.close()
    
    output = open(filename, 'w')
    while not S.is_empty():
        output.write(S.pop() + '\n')
    output.close()

#****************************************************************************
def is_matched(expr):
    """Return True if all delimiters are properly match"""
    lefty = '({['
    righty = ')}]'
    S = ArrayStack()
    for c in expr:
        if c in lefty:
            S.push(c)
        elif c in righty:
            if S.is_empty():
                return False
            if righty.index(c) != lefty.index(S.pop()):
                return False
    return S.is_empty()

def is_matched_html(raw):
    """Return True if all HTML tages are properly match"""
    S = ArrayStack()
    j = raw.find('<')
    while j != -1:
        k = raw.find('>', j+1)
        if k == -1:
            return False
        tag = raw[j+1:k]
        if not tag.startswith('/'):
            S.push(tag)
        else:
            if S is_empty():
                return False
            if tag[1:] != S.pop():
                return False
        j = raw.find('<', k+1)    
    return S.is_empty()    
```

If we use pop(0) and append(e) to implement a LIFO Queue, the pop method would always cost O(n) time since we need to move all the elements leftward. So, __we can replace the dequeued entry in the array with a reference to None, and maintain an explicit variable f to store the index of the element that is currently at the front of the queue.__  

__However there is a drawback__. If we repeatedly enqueue a new element and then dequeue another(allowing the front to drift rightward). The size of the underlying list would grow to O(m) where m is the total number of enqueue operations.  
__Solution:__ Using an Array Circularly.   
__How we do:__ when we dequeue an element and want to "advance" the front index, we use the f = (f + 1)%N. If we have a list of length 10, and a front index 9, we can advance to (9+1) % 10, which is index 0. So we need `_data`, `_size`, `_front` in our queue class.


```python
class ArrayQueue:
    """FIFO queue using python list"""
    DEFAULT_CAPACITY = 10
    
    def __init__(self):
        self._data = [None]*ArrayQueue.DEFAULT_CAPACITY
        self._size = 0
        self._front = 0
    
    def __len__(self):
        return self._size
    
    def is_empty(self):
        return self._size == 0
    
    def first(self):
        if self.is_empty():
            raise Empty('Queue is empty')
        return self._data[self._front]
    
    def dequeue(self):
        if self.is_empty():
            raise Empty('Queue is empty')
        answer = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1)%len(self._data)
        self._size -= 1
        return answer

#*********************************************************************
    def enqueue(self, e):
        if self._size == len(self._data):
            self._resize(2 * len(self.data))
        avail = (self._front + self._size) % len(self._data)
        self._data[avail] = e
        self._size += 1

#*********************************************************************
    def _resize(self, cap):
        old = self._data
        self._data = [None] * cap
        walk = self._front
        for k in range(self._size):
            self._data[k] = old[walk]
            walk = (1 + walk)%len(old)
        self._front = 0
        
# we also need to consider if we want to resize the queue, when we dequeue an element.
```

__Double-ended queue/deque/deck__: A queue-like data structure that supports insertion and deletion at both the front and the back of the queue.  
We can the same three instance variables: `_data`, `_size`, and `_front`. When we need to know the index of the back of the deque, or the first available slot beyond the back of the deque, we can use modular for the computation.  
last() method uses the index `back = (self._front + self._size - 1)%len(self._data)`  
`ArrayDeque.add_last` is the same as `ArrayQueue.enqueue`, including `_resize` utility. `ArrayDequeue.delete_first` is the same as `ArrayQueue.dequeue`. `add_first` and `delete_last` ues similar techniques. One subtlety is `add_fist` may need to wrap around the beginning of the array, so we rely on modular arithmetic to circularly _decrement_ the index, as   
`self._front = (self._front - 1)%len(self._data)`
