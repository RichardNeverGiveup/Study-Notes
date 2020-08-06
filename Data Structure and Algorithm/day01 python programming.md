
## 1. What is Overloading?
x + y calls the _int_ class `__add__` method when x is an integer, but it calls float type 's `__add__`
method when x is a float.  
`+` operator corresponds to `x.__add__(y)`  
The following example shows operator overloading.


```python
class Dog:
    def __init__(self, name, speakText):
        self.name = name
        self.speakText = speakText
        
    def speak(self):
        return self.speakText
    
    def getName(self):
        return self.name
    
    def __add__(self, otherDog):
        return Dog("Puppy of " + self.name + " and " + otherDog.name, self.speakText + otherDog.speakText)

def main():
    boyDog = Dog("Tom", "WOOOOF")
    girlDog = Dog("Lisa", "barkbark")
    puppy = boyDog + girlDog    # this is overloading
    print(puppy.speak())
    print(puppy.getName())

if __name__ == "__main__":
    main()
```

Define `__str__(self)` and `__repr(self)` method in your own class could overload the operator str(x) and repr(x).  
_str_ operator should return a string that is suitable for human interaction  
_repr_ operator is called when a string representation is needed that can be evaluated by _eval_ function


## 2. what is Polymorphism
It means there can be many ways that a particular behavior might be implemented.  
Defining several classes with a same method, but the behavior that particular method implement is different.


```python
class Dog:
    def __init__(self, name):
        self.name = name
    def speak(self):     
        return 'wooof'
    
class Cat:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return 'MiaoMiao'
    
class Robot:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return '%&^*&$%%'
    
def main():
    Tom = Cat("Tom")
    Snoopy = Dog("Snoopy")
    T800 = Robot("T800")
    li = [Tom, Snoopy, T800]
    for i in li:
        # a same speak method, behave differently in different classes
        # this is polymorphism
        print(i.speak())    

if __name__ == "__main__":
    main()
```

    MiaoMiao
    wooof
    %&^*&$%%
    

## 3. how to iterate over our own class
we need define `__iter__(self)` in our class to implement behavior like `for i in ourClassInstance`


```python
class Mylist:
    def __init__(self):
        self.items = []
    
    def append(self, item):
        self.items = self.items + [item]
        
    def __iter__(self):
        # take advantage of the __iter__ funciton of python list
        for c in self.items:
            yield c
mylist = Mylist()
mylist.append('tom')
mylist.append('jerry')
for i in mylist:
    print(i)
```

## 4. XML files
eXtensible Markup Language(XML) is used for describing data in a standard form.  
By using XML, we could eliminate some of the dependence between the Data and Program.

XML document begins with a special line to identify it as an XML file:  
`<?xml version="1.0" encoding="UTF-8" standalone="no" ?>`  
The _minidom_ parser is a XML parser.  
`import xml.dom.minidom`   
`xmldoc = xml.dom.minidom.parse(filename)`  
`graphicsCommands = xmldoc.getElementsByTagName("GraphicsCommands")` would return a list of all elements that match this tag name.
