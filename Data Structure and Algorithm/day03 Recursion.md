
## 0. Warming up
#### Imperative programming
This simply means that you are likely used to thinking about creating variables, storing values, and updating those values as a program proceeds.
#### Functional programming 
When you program in the functional style, you think much more about the definition of __what__ you are programming than __how__ you are going to program it.  
For functional programming, we try not use __for__ or __while__ loop, if you use either kind of loo you are trying to write a function imperatively rather than functionally.  

## 1. Scope
`x = 6` in python means x is a reference that points to an object with a 6 inside it.  
Scope refers to a part of a program where a collection of identifiers are visible. 

When you reference an identifier in a statement in your program,
Python first examines the local scope to see if the identifier is defined there, within
the local scope. An identifier, id, is defined under one of three conditions:  
<ul>
<li>A statement like id = '6' appears somewhere within the current scope.</li>
<li>id appears as a parameter name of the function in the current scope.</li>
<li>id appears as a name of a function or class through the use of a function def or class definition within the current scope</li>
</ul>  

### Enclosing Scope
If Python does not find the reference id within the local scope, it will examine the Enclosing scope to see if it can find id there.  
Notice that function names are included as identifiers.  
The final enclosing scope of a module is the module itself. Each module has its own scope.  

Each scope has its own copy of identifiers. The choice of which val is visible is made by always selecting the innermost scope that defines the identifier.  
If we use an identifier that is already defined in an outer scope, we will no longer be able to access it from an inner scope where the same identifier is defined.  

Every function definition (including the definition of methods) within a module defines a different scope. This scope never includes the function name itself, but includes its parameters and the body of the function.  
### LEGB 
stands for Local, Enclosing, Global, and Built-In, python would look for an id in this order: L>E>G>B  

## 2. Run-Time Stack and Heap
Python splits the RAM up into two parts called the Run-time Stack and the Heap.  
The run-time stack is a stack of Activation Records. The Python interpreter __pushes__ an activation record onto the stack when a function is called and __pops__ the corresponding activation record off the stack when a function returns.  
The __Heap__ in RAM is where all objects are stored. References to objects are stored within the run-time stack and those references point to objects in the heap.  

## 3. Recursive Function
When writing recursive functions we want to think more about what it does than how itworks. There are four rules that you adhere to:  
1. Decide on the name of your function and the arguments that must be passed to it to complete its work as well as what value the function should return.
2. Write the base case for your recursive function first. The base case is an if statement that handles a very simple case in the recursive function by returning a value.
3. Call the function recursively with arguments that are getting smaller and approaching the base case.
4. Pick some values to try out with your recursive function.  


```python
def sumFirstN(n):
    return n*(n+1)//2
def main():
    x = int(input("Please enter a non-negative integer:"))
    s = sumFirstN(x)
    print(s)
if __name__ == "__main__":
    main()
```


```python
def recSumFirstN(n):
    if n == 0:
        return 0
    return recSumFirstN(n-1) + n
```

If we write reverse list function non-recursively, it would be as follow:


```python
def revList(lst):
    accumulator = []
    for x in lst:
        accumulator = [x] + accumulator
    return accumulator
revList([1,2,3,4])
```




    [4, 3, 2, 1]



If we wanna it recursively, fist consider the base case, how to reverse a very simple list, say the empty list.  
Then think about how to make a normal list smaller? Finally, how to piece our solution together?


```python
def revList(lst):
    if lst == []:
        return []
    smallerrev = revList(lst[1:])
    first = lst[0:1]
    result = smallerrev + first
    return result
```

It is possible to make a list or string smaller withoutactually making it physically smaller.  
In that case it may be helpful to write a function that calls a helper function to do the recursion.  
Helper function called revListHelper would do the actual recursion. The list itself does not get smaller in the helper function. Instead, the index argument gets smaller, counting down to 鈭? when the empty list is returned.


```python
def revList2(lst):
    
    def revListHelper(index):
        if index == -1
            return []
        restrev = revListHelper(index-1)
        first = [lst[index]]
        result = first + restrev
        return result
    return revListHelper(len(lst)-1)
```

This example could be rewritten so the index grows toward the length of the list. In that case the distance between the index and the length of the list is the value that would get smaller on each recursive call.

## 4. Type Reflection
Python has another very nice feature called reflection. Reflection refers to the ability for code to be able to examine attributes about objects and the examined result might be used as a function or passed as parameters to a function.  
Like `type(obj)` will return an object which represents the type of obj. we could use this property:  
If obj is a reference to a string, python will return the __str type__ object. Further, we can write __str()__ to get an empty string. like this `type('a')()`.  
Using reflection, we can write one recursive reverse function that will work for strings, lists, and __any other sequence that supports slicing and concatenation.__


```python
def reverse(seq):
    SeqType = type(seq)
    emptySeq = SeqType()
    
    if seq == emptySeq:
        return emptySeq
    restrev = reverse(seq[1:])
    first = seq[0:1]
    
    result = restrev + first
    return result
# this function may be rewriten to the helper function style.
# if the seq supports index operation.
```
