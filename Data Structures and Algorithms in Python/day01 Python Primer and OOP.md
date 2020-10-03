
## Python Primer

Python identifier is like a pointer variable in C++, each identifier is associated with the memory address of the object to which it refers.  
A class is immutable means each object of that class has a fixed value upon instantiation that cannot subsequently be changed. list, set, dict are mutable.  
List is a referential structure, it stores a sequence of references to its items.


```python
def sqrt(x):
    if not isinstance(x, (int, float)):
        raise TypeError('x must be numeric')
    elif x < 0:
        raise ValueError('x cannot be negative')
```


```python
# import collections  
from collections.abc import Iterable
def sum1(values):
#     if not isinstance(values, collections.Iterable):  # this is deprecated in python 3.8
    if not isinstance(values, Iterable):
        raise TypeError('params must be an iterable type')
    total = 0
    for v in values:
        if not isinstance(v, (int, float)):
            raise TypeError('elements must be numeric')
        total = total + v
    return total

sum1((1,2,3))

def sum2(values):
    total = 0
    for v in values:
        total = total + v
    return total
# sum1 and sum2 both works, if values is not an iterable, the for-loop would raise error. if user sends nonnumerical element
# total + v would raise error since total is initialized to 0.
```


```python
try:
    fp = open('sample.txt')
except IOError as e:
    print(e)
    print('Unable to open the file:', e)
```


```python
age = -1
while age < 0:
    try:
        age = int(input('Enter your age in years: '))
        if age <= 0:
            print('Your age must be positive')
    except ValueError:
        print('That is an invalid age specification')
    except EOFError:
        print('There was an unexpected error reading input.')
        raise   # raise the same exception that is currently being handled  # pass is another way

# finally clause with a body of code that will always be executed in the standard or exception cases.
```

__Iterator__ is an object that manages an iteration through a series of values, if i identifies an iterator object, each call to next(i) produces a subsequent element, with StopIteration exception raised to indicate there are no more elements.  
__Iterable__ is an obj, that produces an iterator via iter(obj).  
data  = [1,2,3,4], but call next(data) is illegal, but i = iter(data) and then use next(i) would work. So, list is iterable.  

Calling iter(data) on a list instance produces an instance of the list_iterator class. That iterator does not store its own copy of the elements in the list. It maintains a current index into the original list.  

Implicit iterable: without storing all of its values at once. Like the range(1000) does not return a list of numbers, it returns a range obj that is iterable. This obj generate the values one at a time only as needed, this is __lazy evaluation__.  

The most easy way to create an iterator is through the use of __generators__. It is much like a function, but use __yield__ keyword.


```python
def factors1(n):
    results = []
    for k in range(1, n+1):
        if n % k == 0:
            results.append(k)
    return results
#### this is the generator version ######
def factors2(n):
    for k in range(1, n+1):
        if n % k == 0:
            yield k
# we can use multiple yield in a generator
def factors3(n):
    k = 1
    while k * k < n:
        if n % k == 0:
            yield k
            yield n // k
        k += 1
    if k * k == n:
        yield k

#  infinite Fibonacci series
def fibonacci():
    a = 0
    b = 1
    while True:
        yield a
        # a, b = b, a+b  # concise way
        future = a + b
        a = b
        b = future
```

Conditional Expressions: __expr1 if condition else expr2__, like `param = n if n >= 0 else -n`  
Comprehension syntax: [ expression for v in iterable if condition ]:  
`[k*k for k in range(1, n+1)]` list  
`{k*k for k in range(1, n+1)}` set  
`(k*k for k in range(1, n+1))` generator  
`{k : k*k for k in range(1, n+1)}` dictionary

Scopes: calls to dir() and vars() report on the most locally enclosing namespace in which they are executed.  
First-Class Obj: are instances of a type that can be assigned to an identifier, passed as a parameter, or returned by a function.  
In python, functions and classes are First-Class obj.

## OOP
Modularity: like a house can be viewed as consisting of several interacting units, electrical, heating and cooling, plumbing, and structure  

Abstraction: focus on what each operation does, not how i does it. __Duck typing__ means there is no 'compile time' checking of data types, so there is no formal requirement for declarations of abstract base classes, programmers assume that an obj supports a set of known behaviors, if those assumptions fail interpreter would raise run-time error. So if this obj supports those behaviors, then it is a 'duck'.  

Encapsulation: set free programmers not to worry about writing code of new class that may intricately depends on those internal decisions in the classes they inherited.  

__Design Patterns__:
1. algorithms design patterns: recursion, amortization, divide-and-conquer, prune-and-search, brute force, dynamic programming, the greedy method.  
2. software engineering design patterns: iterator, adapter, position, composition, template method, locator, factory method.  

Software development's major steps: 1.design 2.implementation 3.testing and debugging  
For OOP, there are some rule of thumb that we can use when determing how to design our classes:  
Responsibilities: divide the work into different actors with different responsibilities, use action verb to describe responsibilities. These actors are those classes for us.  
Independence: guarantee each class has autonomy over some aspect of the program.  
Behaviors: define the behaviors for each class precisely so the results of each action performed by a class will be understood by other classes interact with it. These behaviors will define the methods in this class, and then these methods for a class are the interface to the class.  

__CRC cards__: class responsibility collaborator cards. __UML__: unified modeling language.  
the following is a creditcard example based on this design:
<ul>
<li>Class: CreditCard</li>
<li>Fields: _customer, _balance, _bank, _limit, _account</li>
<li>Behaviors: get_customer(), get_balance(), get_bank(), get_limit(), get_account(), charge(price), make_payment(amount)</li>
</ul>  

1. Classes should use singular noun and capitalized name.
2. Functions should be lowercase
3. identifiers for constant should use all capital letters


```python
class CreditCard:
    """A consumer credit card."""
    def __init__(self, customer, bank, acnt, limit):
        """Creat a new credit card instance.
        
        The initial balance is zero.
        
        customer the name of the customer (e.g., 'Ruilin Cheng')
        bank     the name of the bank (e.g., 'California Savings')
        acnt     the acount identifier (e.g., '5391 0375 9387 5309')
        limit    credit limit (measured in dollars)
        """
        self._customer = customer
        self._bank = bank
        self._account = acnt
        self._limit = limit
        self._balance = 0
    
    def get_customer(self):
        """Return name of the customer."""
        return self._customer
    
    def get_bank(self):
        """Return the bank's name."""
        return self._bank
    
    def get_account(self):
        """Return the card identifying number (typically stored as a string)."""
        return self._account
    
    def get_limit(self):
        """Return current credit limit"""
        return self._limit
    
    def get_balance(self):
        """Return current balance"""
        return self._balance
    
    def charge(self, price):
        """Charge given price to the card, assuming sufficient credit limit.
        
        Return True if charge was processed; False if charge was denied.
        """
        if price + self._balance > self._limit:
            return False
        else:
            self._balance += price
            return True
        
    def make_payment(self, amount):
        """Process customer payment that reduces balance."""
        self._balance -= amount
```

__Operator Overloading__:
By default, the + operator is undefined for a new class, but we could provide this definition for the new class by implementing a specially named method `__add__`.   

__Personal notes__: This is a reflection of the 'duck type', the original author of + operator would never know what other programmers would need for + in the future, so they defined a special method for +, if other programmers in the future implemented this method in their own class, then + would be supported. 'maybe new class is a bird, but it walks like a duck, then we treat is as a duck!'  

`__mul__ and __rmul__`: for a * b would call for `a.__mul__(b)`, if the class of a does not implement such method, then python checks the class definition for the right-hand operand b, see if it support `__rmul__`.  
The distinction between `__mul__ and _rmul__` allows a class to define different semantics in cases, such as matrix multiplication.    

__Non-operator overloads__: like `str(foo)`, is formally a call to the constructor for the string class. If the parameter is an instance of a user-defined class, the original author of the string class could not have known how that instance should be portrayed. So the string constructor calls `foo.__str__()`.  
Similar special methods are used to determine how to construct int, float, bool based on a parameter from a user-defined class. Like `if foo:` can even be used when foo is not formally a Boolean value, for a user-defined class, the if condition is evaluated by the `foo.__bool__():`  

__Implied Methods__: generally, if a user-defined class does not implement a particular special method, then the corresponding operator syntax will raise an exception, like `a+b` for instances of a user-defined class without `__add__ or __radd__` will raise an error.  
However, there are some operators that have default definitions provided by python, if the corresponding special methods are not defined, some operators would use definitions from others.  
eg., the `__bool__` method supports `if foo:`, has default semantics so every object other than None is evaluated as True. However if the `__len__` method exists, like for container types, this method normally is used to return the size of the container, then evaluation of `bool(foo)` is interpreted by default to be True for instances with nonzero length.  

we use `__iter__` for iterators, but if a container class provides implementations for both `__len__ and __getitem__`, a default iteration is provided automatically. Furthermore, once an iterator is defined, default functionality of `__contains__` is provided.  
if `__eq__` is not provided then `a == b` becomes `a is b`.  
watch out! some nature implications are not automatically provided by python. like `__eq__` support `a==b` does not mean `a != b` is also defined, acutally we need `__ne__` to provide `a != b`. And providing `__lt__ and __eq__` does not imply semantics for a <= b.


```python
v =  Vector(5) # construct a five-dimensional <0,0,0,0,0>
v[1] = 23      #<0,23,0,0,0> based on use of __setitem__
v[-1] = 45     #<0,23,0,0,45> also via __setitem__
print(v[4])    #print 45 via __getitem__
u = v + v      #<0,46,0,0,90> via __add__
total = 0
for entry in v:  # implicit iteration via __len__ and __getitem__
    total += entry   
```

eg: if we want to implement a Vector Class with this above attributes. What should we do?  
Note that, our Vector class automatically supports the syntax u = v + [5,3,10,-2,1]. We never declared the other parameter of our `__add__` method as another Vector instance. The only behaviors we rely on for paramter other is that it supports len(other) and access to other[j]. This is the __polymorphism__ of python, the base form is that two methods, if your user-defined special class supports these two methods, then we can use its instance as operand(__we treat it as a duck, if it walks and swims like a duck__)


```python
class Vector:
    """Represent a vector in a multidimensional space."""
    def __init__(self,d):
        """Creat d-dimensional vector of zeros."""
        self._coords = [0] * d
    def __len__(self):
        """Return the dimension of the vector"""
        return len(self._coords)
    def __getitem__(self,j):
        """Return jth coordinate of vector"""
        return self._coords[j]
    def __setitem__(self,j,val):
        """Set jth coordinate of vector to a val"""
        self._coords[j] = val
    def __add__(self, other):
        """Return sum of two vectors"""
        if len(self) != len(other):
            raise ValueError('dimensions must be the same')
        result = Vector(len(self))
        for j in range(len(self)):
            result[j] = self[j] + other[j]
        return result
    def __eq__(self,other):
        return self._coords == other._coords
    def __ne__(self,other):
        return not self == other
    def __str__(self):
        return '<' + str(self._coords)[1:-1] + '>'

# for now, this class constructor do not support passing in a sequence as params.
```

__iterator__: supports `__next__` method to return the next element for the collection or rasise a StopIteration exception to indicate there are no more elements. We seldom need to implement an iterator class directly, the preferred way is use __generator__ syntax through the `yield` keyword.  

python supports automatic iterator implementation for a class that defines both `__len__ and __getitem__`.


```python
class SequenceIterator:
    """An iterator for any of Python's sequence types."""
    def __init__(self, sequence):
        """Create an iterator for the given sequence."""
        self._seq = sequence         # keep a reference to the underlying data
        self._k = -1
        
    def __next__(self):
        """Return the next element, or else raise StopIteration error."""
        self._k += 1
        if self._k < len(self._seq):
            return(self._seq[self._k])
        else:
            raise StopIteration
    
    def __iter__(self):
        """By convention, an iterator must return itself as an iterator"""
        return self
```


```python
class Range:
    """A class that mimics the built-in range class."""
    def __init__(self,start,stop=None,step=1):
        if step == 0:
            raise ValueError('step cannot be 0')
        if stop is None:
            start, stop = 0, start
        
        # calculate the effective length once
        self._length = max(0, (stop - start + step - 1)//step)
        
        self._start = start
        self._step = step
        
    def __len__(self):
        return self._length
    
    def __getitem__(self,k):
        if k < 0:
            k += len(self)
        if not 0 <= k < self._length:
            raise IndexError('index out of range')
        return self._start + k*self._step
```


```python
class PredatoryCreditCard(CreditCard):
    """An extension to CreditCard that compounds interest and fees."""
    def __init__(self,customer,bank,acnt,limit,apr):
        """Creat a new predatory credit card instance.
        
        The initial balance is zero.
        
        customer the name of the customer (e.g., 'Ruilin Cheng')
        bank     the name of the bank (e.g., 'California Savings')
        acnt     the acount identifier (e.g., '5391 0375 9387 5309')
        limit    credit limit (measured in dollars)
        apr      annual percentage rate(eg., 0.0825 for 8.25% APR)
        """
        super().__init__(customer,bank,acnt,limit)
        self._apr = apr
    
    def charge(self, price):
        """Charge given price to the card, assuming sufficient credit limit.
        
        Return True if charge was processed.
        Return False and assess $5 fee if charge is denied.
        """
        success = super().charge(price)
        if not success:
            self._balance += 5
        return success
    
    def process_month(self):
        """Assess monthly interest on outstanding balance."""
        if self._balance > 0:
            monthly_factor = pow(1 + self._apr, 1/12)
            self._balance *= monthly_factor
```

PredatoryCreditCard subclass directly accesses the data memeber `self._balance`, which was established by the parent CreditCard class. The underscore name, by convention should not be accessed this way, but subclass has privilege to do so. For C++, for nonpublic members, should be declared with __protected__(single underscore in python) or __priviate__(double underscore in python). Protected members are accessbile to subclass, Priviate members should not be accessible both to public or subclass.  

In the above subclass, we created dependency since we used this nonpublic member `self._balance`, if the CreditCard changed its internal design, our class may lost its functionality.  
We could use public get_balance() method to retrieve the current balance within the process_month() method, but the current design of CreditCard dose not afford an effective way for subclass to change the balance, other than by direct manipulation of the data member.  
If we can redesign the CreditCard, we can add a nonpublic method, `_set_balance`, that could be used by subclass to change balance without directly accessing the data member `_balance`. And if the author of CreditCard want to redesign this class, he would also leave this interface for the potential subclass defined by others in the future. So, in this way, subclass definers will not worried about the dependency that the baseclass internal data member may change some day.
```python
class Progression:
    """Iterator producing a generica progression.
    
    Default iterator produces the 0,1,2,...
    """
    
    def __init__(self, start=0):
        """Initialize current to the first value of the progression."""
        self._current = start
        
    def _advance(self):
        """Update self._current to a new value.
        
        This is provided for subclass, and should be overridden to customize progression.
        
        By convention, if current is set to None, this designates the end of a finite progression.
        """
        self._current += 1
    
    def __next__(self):
        """Return the next element or Raise StopIteration error."""
        if self._current is None:
            raise StopIteration()
        else:
            answer = self._current
            self._advance()
            return answer
    
    def __iter__(self):
        """By convention, an iterator must return itself as an iterator."""
        return self
    
    def print_progression(self, n):
        """Print next n values of the progression."""
        print(' '.join(str(next(self)) for j in range(n)))
```


```python
class ArithmeticProgression(Progression):
    """Iterator producing an arithmetic progression."""
    
    def __init__(self, increment=1, start=0):
        """Create a new arithmetic progression.
        
        increment       the fixed constant to add to each term(default 1)
        start           the first term of the progression(default 0)
        """
        super().__init__(start)
        self._increment = increment
        
    def _advance(self):
        self._current += self._increment
```


```python
class GeometricProgression(Progression):
    def __init__(self, base=2, start=1):
        """
        base      the fixed constant multiplied to each term(default 2)
        start     the first term of the progress(default 1)
        """
        super().__init__(start)
        self._base = base
        
    def _advance(self):
        self._current *= self._base
```


```python
class FibonacciProgression(Progression):
    def __init__(self, first=0, second=1):
        super().__init__(first)
        self._prev = second - first  # fictitious value preceding the first
    
    def _advance(self):
        self._prev, self._current = self._current, self._prev + self._current  # here is why not just set _prev to a causal value.
        # self._prev = second - first is useful when poping the second element.
```

In OOP, __abstract base class__ is only for serving as a base class through inheritance and it can not be directly instantiated. While __concret class__ is one that can be instantiated. So our Progression class is technically concrete but we essentially designed it as an ABC.  
In statically typed language like C++, an ABC serves as a formal type that may guarantee one or more __abstract methods__. This provides support for __polymorphism__, _as a variable may have an ABC as its declared type, even though it refers to an instance of a concrete subclass_.  

Although python has no declared type and there is no strong tradition of defining ABCs in python, but Python's collections module provides several ABCs that are useful when we want to define custom data structure that share common interface with built-in data structures.  

These rely on __template method pattern__. This design pattern is when an ABC provides concrete behaviors that rely upon calls to other abstract behaviors. As soon as a subclass of this ABC provides definitions for the missing abstract behaviors, the inherited concrete behaviors are well defined.  

Eg., the collections.Sequence ABC defines behaviors common to python's list, str, and tuple, as sequences that support element access via an integer index(so the `__getitem__ and __len__`method are abstract methods in this ABC). This ABC provides concrete implementation of methods, count, index, and `__contains__` that can be inherited by any class that provides concrete implementations of both `__len__ and __getitem__`.


```python
from abc import ABCMeta, abstractmethod
class Sequence(metaclass=ABCMeta):
    """Our own version of collections.Sequence abstract base class."""
    @abstractmethod
    def __len__(self):
        """Return the length of the sequence."""
    
    @abstractmethod
    def __getitem__(self, j):
        """Return the element at index j of the sequence."""
        
    def __contains__(self, j):
        """Return True if val found in the sequence."""
        for j in range(len(self)):
            if self[j] == val:
                return True
        return False
    
    def index(self, val):
        """Return leftmost index at which val is found"""
        for j in range(len(self)):
            if self[j] == val:
                return j
        raise ValueError('value not found.')
        
    def count(self, val):
        """Return the number of elements equal to given value."""
        k = 0
        for j in range(len(self)):
            if self[j] == val:
                k += 1
        return k
```
