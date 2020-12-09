#!/usr/bin/env python
# coding: utf-8

# Python is __Dynamically typed__ language, there is no advance declaration associating an indentifier with a particular data type. An identifier can be associated with any type of object(or reassigned to another object of different type). C++ use type to allocate memory for an identifier, this can save space and be more efficient.  
# Q1: _But in python, everything is object, how this dynamic memory allocation is implemented? and how to make it as efficient as possible. How the interpreter do the dirty work for us, like determine how much space for a "3.26" we passed in? And what is the difference between python obj construct and the heap memory for (new obj) in C++._   
# _Guess: The memory is determined by the right operand of an assigment, the identifier(left operand) contains the start and end memory address of the right operand. Not like C++ (only the starting address)._ 
# 
# List, set, dict are mutable bulit-in classes, but there are some subtles in the low-level implementation.  
# Q2: _what data structure concepts are used in the C level for these built-in container? Why should the key of a dict keep immutable?_  
# 
# Bitwise Operators:  
# 1.~ bitwise complement(prefix unary operator)  
# 2.& biwise and; | bitwise or; ^ bitwise exclusive-or;  
# 3.<< shift bits left, filling in with zeros; >> shift bits right, filling in with sign bit  

# In[ ]:


def sqrt(x):
    if not isinstance(x, (int, float)):
        raise TypeError('x must be numeric')
    elif x < 0:
        raise ValueError('x cannot be negative')

age = -1
while age <= 0:
    try:
        age = int(input('Enter your age in years: '))
        if age <= 0:
            print('Your age must be positive')
    except ValueError:
        print('That is an invalid age specification')
    except EOFError:
        print('There was an unexpeceted error reading input.')
        raise
    finally:
        print('This sentence will always come up')


# An Iterable obj means this obj can produce an iterator via `iter(obj)`.  
# Iterator is an obj that manages an iteration and we can use bulit-in function `next(i)` to produce a subsequent element, with a StopIteration exception raised when there is no more element.  
# eg: li = [1,2,3,4], list is iterable but we can not call `next(li)` directly, we can use `i = iter(li)` to get a iterator first, then call `next(i)` to get element in the list one by one.  
# Iterator does not store its owen copy of the list of elements, it keeps a current index into the original list.  
# 
# Generator is one simple way of making iterators, its syntax is similar to a function, but ues `yield` to replace `return`.  
# like the following code, we can use `for factor in factors(100)` to loop this iterable obj.

# In[ ]:


def factors(n):
    print('normal factor function is called')
    for k in range(1, n+1):
        if n % k == 0:
            yield k
            
def factors(n, dummy):
    print('overload factor function with dummy is called')
    k = 1
    while k * k < n:
        if n % k == 0:
            yield k
            yield n // k
        k += 1
    if k * k == n:
        yield k
    
for f in factors(10):
    print(f)
# BUG 1, if we put def factors(n, dummy) after def factors(n), and call factor(10), this would raise exception.
# this is a difference between C++ and python. In python, function can't overload by changing the number of params we pass in. 

def fibonacci():
    a = 0
    b = 1
    while True:
        yield a
        future = a + b
        a = b
        b = future

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a+b

[ k*k for k in range(1, n+1) ]
{ k*k for k in range(1, n+1) } # set comprehension
( k*k for k in range(1, n+1) ) # generator comprehension
{ k : k*k for k in range(1, n+1) } # dictionary comprehension


# use dir() and vars() to get the most locally enclosing namespace in which they are executed.  
# _First class obj_ : instances of a type that can be _assigned to an identifier, passed as a parameter, or returned by a function_. 

# In[ ]:


# a demo for Scope of python function

p1, p2 = "yeye","nainai"
def globalfun():
    g_p1 = "baba"
    g_p2 = "mama"
    
#     local()    # 1.this would cause error, use it after function defination
    def local():
        print("-----by local--------")
        print("p1 and p2:")
        print(p1, p2)
        print("g_p1 and g_p2:")
        print(g_p1, g_p2)
        print("o_p1 and o_p2:")
#         print(o_p1, o_p2)    #  2. this is bug2, local fun can not get params in outer fun
        print("-----by local--------")
    local()  # 1.use a fun after function defination can fix bug 1
    outer()
    
def outer():
    o_p1 = "shushu"
    o_p2 = "ayi"
    print("-----by outer--------")
    print("g_p1 and g_p2:")
#     print(g_p1, g_p2)   # this is bug3, outer can not access params in local fun
    print("l_p1 and l_p2:")
#     print(l_p1, l_p2)    # this is same as bug3
    print("p1 and p2:")
    print(p1, p2)       
    print("-----by outer--------")

globalfun()


# Software development  
# __Responsibilities__(divide the work into different actors, use action verbs to describe responsibilities);
# __Independence__;
# __Behaviors__  

# In[49]:


class CreditCard:
    """A consumer credit card."""
    def __init__(self, customer, bank, acnt, limit):
        """Create a new credit card instance.
        
        The initial balance is zero.
        
        customer    the name of the customer(e.g., 'Ray Cheng')
        bank        the name of the bank(e.g., 'ICBC')
        acnt        the acount ID(e.g., '5391 0375 5301')
        limit       credit limit(measured in dollars)
        """
        self._customer = customer
        self._bank = bank
        self._account = acnt
        self._limit = limit
        self._balance = 0
        
    def get_customer(self):
        return self._customer
    def get_bank(self):
        return self._bank
    def get_account(self):
        return self._account
    def get_limit(self):
        return self._limit
    def get_balance(self):
        return self._balance
        
    def charge(self, price):
        if price + self._balance > self._limit:
            return False
        else:
            self._balance += price
            return True
    def make_payment(self, amount):
        self._balance -= amount


# Operator Overloading: by default, the + is undefined for a new class, but we can provide a definition of + for this class.  
# Providing `__add__` and `__mul__` in our new class to define its behavior.  
# Non-Operator Overloads: i is an instance of our new class, if we want to present it in string like we did in `str(9)`(the str representation of a number). we will use `str(i)`, but by default this builit-in func doesn't support our new class. So we can define `__str__()` in our class. `__bool__` is similar.  
# 
# ### Implied Methods  
# Q1: How these implied methods implemented?   
# Guess: define concrete method(rely on abstract method) in the ABC, then implement the abstract method in subclass. So if we define a new class with several special method defined, then the concrete method in ABC inherited by our new class would automatically be supported.  
# 
# `__bool__` method supports `if foo:`, has default semantic, every obj other than None is evaluated as True. But for container types, if `__len__` is supported in the container, then `bool(container)` is to be True when the container instance is nonzero length.  
# 
# `__iter__` can provide iterators for collections. But for container class, if `__len__` and `__getitem__` are implemented, then a default iteration is provided. And once an iterator is defined, then `__contains__` is provided.  
# If `__eq__` is not defined in a new class, then `a==b` will be evaluated as `a is b`.  
# 
# __polymorphism__ in python: if we use `u = v + [1,2,3]`, it would result in the element-by-element sum of the Vector and the list. This polymorphism is subtle compared with C++.

# In[43]:


class Vector:
    def __init__(self, d):
        self._coords = [0] * d
        
    def __len__(self):
        """Return the dimension of the vector"""
        return len(self._coords)
    
    def __getitem__(self, j):
        """Return the jth coordinate of vector."""
        return self._coords[j]
    
    def __setitem__(self, j, val):
        self._coords[j] = val
        
    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError('dimensions must agree')
        result = Vector(len(self))
        for j in range(len(self)):
            result[j] = self[j] + other[j]
        return result
    
    def __eq__(self, other):
        return self._coords == other._coords
    
    def __ne__(self, other):
        return not self == other
    
    def __str__(self):
        return '<' + str(self._coords)[1:-1] + '>'


# In[ ]:


class SequenceIterator:
    """An iterator for any of Python's sequence types."""
    def __init__(self, sequence):
        """Create an iterator for the given sequence."""
        self._seq = sequence    # keep a reference to the underlying data
        self._k = -1           # will increment to 0 on first call to next
    
    def __next__(self):
        """Return the next element, or else raise StopIteration error."""
        self._k += 1
        if self._k < len(self._seq):
            return(self._seq[self._k])
        else:
            raise StopIteration()
    
    def __iter__(self):
        """By convention, an iterator must return itself as an iterator."""
        return self


# In[ ]:


class Range:
    """A class that mimics the built-in range class."""
    def __init__(self, start, stop=None, step=1):
        if step == 0:
            raise ValueError('step cannot be 0')
        if stop is None:
            start, stop = 0, start
        
        self._length = max(0, (stop-start+step-1)//step)
        self._start = start
        self._step = step
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, k):
        if k < 0:
            k += len(self)
        if not 0 <= k < self._length:
            raise IndexError('index out of range')
        return self._start + k*self._step
Range(10)[9]


# In[50]:


class PredatoryCreditCard(CreditCard):
    def __init__(self, customer, bank, acnt, limit, apr):
        super().__init__(customer, bank, acnt, limit)
        self._apr = apr
    def charge(self, price):
        success = super().charge(price)
        if not success:
            self._balance += 5
        return success
    def process_month(self):
        """Assess monthly interest on outstanding balance."""
        if self._balance > 0:
            monthly_factor = pow(1 + self._apr, 1/12)
            self._balance *= monthly_factor
# this subclass would rely on the _balance, single underscore means protected(double underscore means private) 
# and can be access by subclass, the superclass may change, so it is better to define a nonpublic method in the superclass
# like _set_balance.


# In[53]:


class Progression:
    """Iterator producing a generic progression.
    Default iterator produces the whole numbers 0,1,2,...
    """
    def __init__(self, start=0):
        self._current = start
    def _advance(self):
        """Update self._current to a new value.
        
        This should be overridden by a subclass to customize progression.
        
        By convention, if current is set to None, this designates the end of a finite progression.
        """
        self._current += 1
        
    def __next__(self):
        if self._current is None:
            raise StopIteration()
        else:
            answer = self._current
            self._advance()
            return answer
    
    def __iter__(self):
        return self
    def print_progression(self, n):
        print(' '.join(str(next(self)) for j in range(n)))

class ArithmeticProgression(Progression):
    def __init__(self, increment=1, start=0):
        """Create a new arithmetic progression.
        
        increment  the fixed constant to add to each term
        start      the first term of the progression
        """
        super().__init__(start)
        self._increment = increment
        
    def _advance(self):
        self._current += self._increment

class GeometricProgression(Progression):
    def __init__(self, base=2, start=1):
        super().__init__(start)
        self._base = base
    def _advance(self):
        self._current *= self._base

class FibonacciProgression(Progression):
    def __init__(self, first=0, second=1):
        super().__init__(first)
        self._prev = second - first
        
    def _advance(self):
        self._prev, self._current = self._current, self._prev + self._current


# __Template method pattern__ is when an ABC provides concrete behaviors that rely upon calls to other abstract behaviors. In that way, as soon as a subclass provides definitions for the missing abstract behaviors, the inherited concrete behaviors are well defined.  
# collections.Sequence class provides concrete implementations of methods, `count, index, __contains__` that can be inherited by any class that provides concrete implementations of both `__len__ and __getitem__`.

# In[54]:


# demo for template method pattern

from abc import ABCMeta, abstractmethod

class Sequence(metaclass = ABCMeta):
    """Our own version of collections.Sequence abstract base class."""
    
    @abstractmethod
    def __len__(self):
        """Return the length of the sequence."""
    @abstractmethod
    def __getitem(self, j):
        """Return the element at index j of the sequence."""
        
    def __contains__(self, val):
        for j in range(len(self)):
            if self[j] == val:
                return True
        return False
    
    def index(self, val):
        for j in range(len(self)):
            if self[j] == val:
                return j
        raise ValueError('value not in sequence')
    
    def count(self, val):
        k = 0
        for j in range(len(self)):
            if self[j] == val:
                k += 1
        return k


# In[ ]:


# demo for deep copy and shallow copy

import copy

red = [1,1,1]
green = [2,2,2]
blue = [3,3,3]
orgin = [red, green, blue]
new = list(orgin)    # shallow copy
print(orgin)
print(new)
new[0][0] = 'a'   # orgin and new all changed
print(orgin)    
print(new)

new2 = copy.deepcopy(orgin)
new2[0][1] = 'a'
print(orgin)    
print(new2)

