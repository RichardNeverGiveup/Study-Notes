
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

```
