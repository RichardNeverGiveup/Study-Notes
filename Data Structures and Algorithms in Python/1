```python
from time import time
start_time = time()
# run algorithm
end_time = time()
elapsed = end_time -  start_time
```

Quadratic function: f(n) = n^2. for nested loops: 1st iteration, when the outer loop do one operation, the inner do one operation; 2nd itertation, the outer do one operation, the inner will need to finish two operations........(i=0,j=0; i=1,j=0,1; i=2,j=0,1,2; ....) So, the totall operations is 1+2+3+4+...+n.  

floor function: the largest integer less than or equal to x.  
ceiling function: the smallest integer greater than or equal to x.  

Big-Oh: f(n) is O(g(n)) if there is a real constant c>0 and an integer constant n0 >= 1 such that f(n) <= c * g(n), for n >= n0.  
Big-Omega; Big-Theta;  

Python's list is implemented as __arry-based sequences__, references to a list's elements are stored in a consecutive block of memory, the j-th element can be found by validating the index. Computer hardware supports constant-time access to an element based on its memory address. Therefore, data[j] is evaluated in O(1) time.  

When we want to find the max in a random list, how much time do we need? The probability that the j-th element is the largest of the first j elements is 1/j. Hence the expected number of times we update the biggest is SUM of 1/j(for j from 1 to n). The result should be O(log n).  

Prefix Averages  
Given a sequence S consisting of n numbers, we want to compute a sequence A such that A[j] is the average of the elements S[0],..,S[j], for j = 0,...,n-1  
A[j] = (SUM of S[i], i from 0 to j) / (j+1)  
Application: the year-by-year returns of a fund, ordered from recent to past; A stream of daily web usage logs.


```python
# demo
def prefix_average1(S):
    n = len(S)
    A = [0] * n
    for j in range(n):
        total = 0
        for i in range(j+1):  # range(0) is None, range(1) is 0, range(2) is 0 and 1.
            total += S[i]
        A[j] = total / (j+1)
    return A
# the inner loop is executed 1+2+3+...+n, so this cost O(n^2) time.

def prefix_average2(S):
    n =  len(S)
    A = [0] * n
    for j in range(n):
        A[j] = sum(S[0:j+1]) / (j+1)
    return A
# this seems to be more efficient by using only one for loop, but the sum function takes O(j+1) time to get the sum of the slice.

def prefix_average3(S):
    n = len(S)
    A = [0] * n
    total = 0
    for j in range(n):
        total += S[j]
        A[j] = total / (j+1)
    return A
# this version only takes O(n) time.
```


```python
def disjoint1(A, B, C):
    """Return True if there is no element common to all three lists."""
    for a in A:
        for b in B:
            for c in C:
                if a == b == c:
                    return False
    return True
# 0(n^3)

def disjoint2(A, B, C):
    for a in A:
        for b in B:
            if a == b:  # only check C if we found match from A and B
#-----------------------------------------------------------------------------#
                for c in C:
                    if a == c:
                        return False
    return True
# Once inside the loop over B, if a and b do not match, then we won't need to check c.
# The worst-case for this disjoint2 is O(n^2). There are quadratically(n^2) many pairs (a, b) to consider.
# However, if A, B are each sets of distinct elements, there can be at most O(n) such pairs with a=b
# eg: A(1,2,3,4), B(1,2,3,4), only (1,1),(2,2),(3,3),(4,4)
# before the #---# line, it would execute n^2 time, but only n time would pass the a==b condition
# the test a == b is evaluated 0(n^2) times, The rest of the time depends upon how many matching(a,b) exists.
# there are at most n pairs, so the a == c is evaluated at most O(n^2) times, so the total time spent is O(n^2).
```


```python
def unique1(S):
    """Return True if there are no duplicate elements in sequence S."""
    for j in range(len(S)):
        for k in range(j+1, len(S)):
            if S[j] == S[k]:
                return False
    return True

# use sort as tool, qucik sort is O(n*log n)
def unique2(S):
    temp = sorted(S)
    for j in range(1, len(temp)):
        if S[j-1] == S[j]:
            return False
    return True
```

In python, each time a function is called, a structure named __activation record__ or __frame__ is created to store information about the progress. This record includes a namespace for storing the function call's parameters and local variables, and information about which command in the body of the function is currently executing.


```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
```

If we want to find one element in a sequence, what can we do? When the sequence is unsorted, the standard way is to loop every element. When the sequence is sorted and indexable, we can use binary search. Binary search runs in O(log n) time.  


```python
def binary_search(data, target, low, high):
    """Return True if target is found in indicated portion of a Python list.
    
    The search only considers the portion from data[low] to data[high] inclusive.
    """
    if low > high:
        return False
    else:
        mid = (low + high)//2
        if target == data[mid]:
            return True
        elif target < data[mid]:
            return binary_search(data, target, low, mid - 1)
        else:
            return binary_search(data, target, mid + 1, high)
```


```python
import os

def disk_usage(path):
    """Return the number of bytes used by a file/folder and any descendents."""
    total = os.path.getsize(path)           # return 0, if the path is a folder.
    if os.path.isdir(path):
        for filename in os.listdir(path):
            childpath = os.path.join(path, filename)
            total += disk_usage(childpath)
    print('{0:<7}'.format(total), path)
    return total
```


```python
def bad_fibonacci(n):
    if n <= 1:
        return n
    else:
        return bad_fibonacci(n-2) + bad_fibonacci(n-1)

def good_fibonacci(n):
    """Return pair of Fibonacci numbers, F(n) and F(n-1)."""
    if n <= 1:
        return (n,0)
    else:
        (a,b) = good_fibonacci(n-1)
        return (a+b,a)
```

If a recursive call starts at most one other, we call this a linear recursion.  
If a recursive call starts two others, we call this a binary recursion.  
If a recursive call starts three or more others, we call this a multiple recursion.  
good_fibonacci, factorial, and binary search are __linear recursion__. Binary search includes a case with two branches that lead to recursive calls, but only one of those calls can be reached during a particular execution of the body.  


```python
def linear_sum(S, n):
    """Return the sum of the first n numbers of sequence S."""
    if n == 0:
        return 0
    else:
        return linear_sum(S, n-1) + S[n-1]

def reverse(S, start, stop):
    """Reverse elements in implict slice S[start:stop]."""
    if start < stop - 1:    # "abc"[0:1] get 'a', so when start = stop-1, we get only one element, but we need at least two.
        S[start], S[stop-1] = S[stop-1], S[start]
        reverse(S, start+1, stop-1)
# this reverse function will terminate after a total of 1 + floor(n/2) recursive calls.


def power(x, n):
    if n == 0:
        return 1
    else:
        return x * power(x, n-1)
# this version runs in 0(n) time. However there is a much faster way using a squaring technique.
# let k denote the floor of the division(in pythin it is n//2). we consider the expression (x^k)^2, 
# when n is even,the floor of n/2 is n/2, so (x^k)^2 = x^n.
# when n is odd, the floor of n/2 is (n-1)/2, so (x^k)^2 = x^(n-1), therefore x^n = x*(x^k)^2

def power2(x ,n):
    if n == 0:
        return 1
    else:
        partial = power(x, n//2)
        result = partial * partial
        if n % 2 == 1:
            result *= x
        return result

# we observe that the n in each recursive call is at most half of the preceding n. So our power2 results in 0(log n) time.
# This power2 save space as well, the depth of this function is O(log n).
```


```python
def linear_sum(S, n):
    """Return the sum of the first n numbers of sequence S."""
    if n == 0:
        return 0
    else:
        return linear_sum(S, n-1) + S[n-1]
    
def binary_sum(S, start, stop):
    """Return the sum of the numbers in implicit slice S[start:stop]."""
    if start >= stop:
        return 0
    elif start == stop-1:
        return S[start]
    else:
        mid = (start + stop) // 2
        return binary_sum(S, start, mid) + binary_sum(S, mid, stop)  # slice S[start:mid] would omit index mid get start -> (mid-1)

# the depth of the recursion is 1+logn, so this function uses O(log n) amount of additional space, this is a big improvement 
# compared with the linear_sum. However the running time of binary_sum is O(n), as there are 2n-1 function calls.
```

Parameterizing a Recursion.  
binary_search(data, target, low, high) seems not explicit enough compare with the way like binary_search(data, target). But if we use this cleaner form, the only way to invoke a search on half the list would have been to make a new list. But making a copy of half the list would already take O(n) time.  
If we wished to provide interface like binary_search(data, target) for our user, we can define in its body invoke a nonpublic utility function having the desired recursive parameters.  

If our computer memory is at a premium, it is useful to use nonrecursive algorithms. And we can use stack structure to replace default recursion provided by Python interpreter to minimize memory usage by storing only the minimal information necessary.  

A tail recursion must be a linear recursion, and can be reimplemented by a loop. A recursion is a tail recursion if any recursive call that is made from one context is the very last operation in that context, with the return value of the recursive call immediately returned by the enclosing recursion.


```python
# keep in mind sequence should be sorted and indexable.

def binary_search(data, target, low, high):
    """Return True if target is found in indicated portion of a Python list.
    
    The search only considers the portion from data[low] to data[high] inclusive.
    """
    if low > high:
        return False
    else:
        mid = (low + high)//2
        if target == data[mid]:
            return True
        elif target < data[mid]:
            return binary_search(data, target, low, mid - 1)
        else:
            return binary_search(data, target, mid + 1, high)

        
def binary_search_iterative(data, target):
    low = 0
    high = len(data) - 1   # count from 0, the high must be len - 1
    while low <= high:    # low = high = mid = 3, if t>data[3], then low=4,high=3(Break); if t<data[3], then high=2,low=3(break);
        mid = (low + high) // 2
        if target == data[mid]:
            return True
        elif target < data[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return False
```


```python
def reverse_iterative(S):
    start, stop = 0, len(S)
    while start < stop - 1:
        S[start], S[stop-1] = S[stop-1], S[start]
        start, stop = start+1, stop-1
```
