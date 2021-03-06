
## 0. Node Class
PyList we defined is a randomly accessible list, we can access any element of the list in O(1) time. Appending an item was possible in O(1) time using amortized complexity analysis, but inserting an item took O(n) time where n was the number of items __after the location where the new item was being inserted.__  

If we wants to insert a large number of items towards the beginning of a list, we could use linked list.  
Each link in the linked list chain is called a Node.  
Node consists of an item(the data associated with the node) and a link(the next node, often called next).  

The first node is a __dummy node__. Having one extra dummy node at the beginning of the sequence eliminates a lot of special cases that must be considered when working with the linked list. An empty sequence still has a dummy node.


```python
class Node:
    def __init__(self, item, next=None):
        self.item = item
        self.next = next
    def getItem(self):
        return self.item
    def getNext(self):
        return self.next
    def setItem(self, item):
        self.item = item
    def setNext(self, next):
        self.next = next
```

## 1. LinkedList
We keep a reference to the first node in the sequence so we can traverse the nodes whennecessary. The reference to the last node in the list makes it possible to append an item to the list in O(1) time. We also keep track of the number of items so we dont have to count when someone wants to retrieve the list size.  
So there are three things in a linkedlist:
1. firstnode
2. lastnode
3. numItems


```python
class LinkedList:
    class __Node:
        def __init__(self, item, next=None):
            self.item = item
            self.next = next
        def getItem(self):
            return self.item
        def getNext(self):
            return self.next
        def setItem(self, item):
            self.item = item
        def setNext(self, next):
            self.next = next
            
    def __init__(self, contents=[]):
#         the first and last item both point to a dummy node at the beginning.
#         this dummy node will always be in the first position and will never contain an item.
        self.first = LinkedList.__Node(None, None)
        self.last = self.first
        self.numItems = 0
        
        for e in contents:
#             this append method hasnt implemented yet in this piece of code
            self.append(e)
    
    def __getitem__(self, index):
        if index >= 0 and index < self.numItems:
    #         pass the dummy node
            cursor = self.first.getNext()
            for i in range(index):
                cursor = cursor.getNext()
            return cursor.getItem()
        raise IndexError("index out of range")
    
    def __setitem__(self, index, val):
        if index >= 0 and index < self.numItems:
            cursor = self.first.getNext()
            for i in range(index):
                cursor = cursor.getNext()
            cursor.setItem(val)
            return
        raise IndexError("assignment index out of range")
        
    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError("Concatenate undefined for " + str(type(self)) + " + " + str(type(other)))
        
        result = LinkedList()
        cursor = self.first.getNext()
        
        while cursor != None:
            result.append(cursor.getItem())
            cursor = cursor.getNext()
            
        cursor = other.first.getNext()
        
        while cursor != None:
            result.append(cursor.getItem())
            cursor = cursor.getNext()
        return result
    
    def append(self, item):
#         we create a new node and make the orginal last one point at it.
#         Then we move the "cursor", make the new node the new self.last node,
#         and increment the number of items by 1
        node = LinkedList.__Node(item)
        self.last.setNext(node)
        self.last = node
            
    def insert(self, index, item):
        cursor = self.first # first is dummy node
        
#         index is always smaller than numItems or it would be add one
#         more element in the tail.
        if index < self.numItems:
            for i in range(index):
#                 if we wanna insert at index 2. the for loop would loop twice
#                 the cursor would move to index 1
                cursor = cursor.getNext()
#             if we wanna insert at index 2, now cursor is index 1, then we
#             make the new node's next points to the element at index 2
#             then we set the cursor's next points to the new node
#             by doing so the inserted node is linked forward and backward.
            node = LinkedList.__Node(item, cursor.getNext())
            cursor.setNext(node)
            self.numItems += 1
        else:
            self.append(item)
```

The append operation has a complexity of O(1) for LinkedLists where with lists the complexitywas an amortized O(1) complexity. Each append will always take the same amount of time with a LinkedList. A LinkedList is also never bigger than it has to be. However, LinkedLists take up about twice the space of a randomly accessible list.  

When working with a LinkedList the n is the number of elements that appear before the insertion point because we must search for the correct insertion point. So inserting at the beginning of a list is a O(n) operation, inserting at the beginning of a LinkedList is a O(1) operation.  

Sort operation is not applicable on linked lists. Efficient sorting algorithms require random access to a list. Insertion sort, a O(n2) algorithm.  
__It would be much more efficient to copy the linked list to a randomly accessible list, sort it, and then build a new sorted linked list from the sorted list.__

## 2. Stacks and Queues
### Stacks
Either a list or a linked list will suffice to implement a stack.


```python
class Stack:
    def __init__(self):
        self.items = []
    
    def pop(self):
        if self.isEmpty():
            raise RuntiemError("Attempt to pop an empty stack")
        topIdx = len(self.items) - 1
        item = self.items[topIdx]
        del self.items[topIdx]
        return item
    
    def push(self, item):
        self.items.append(item)
        
    def top(self):
        if self.isEmpty():
            raise RuntiemError("Attempt to get top of empty stack")
        topIdx = len(self.items) - 1
        
        return self.items[topIdx]
    
    def isEmpty(self):
        return len(self.items) == 0
```

### Queues
To implement a queue with best complexities we need to be able to add to one end of a sequence and remove from the other end of the
sequence in O(1) time. This suggests the use of a linked list.  
However, we can use a list if we are willing to accept an amortized complexity of O(1) for the dequeue operation to implement a queue.


```python
class Queue:
    def __init__(self):
        self.items = []
        self.frontIdx = 0
    
    def __compress(self):
        # this function is to compress the queue when the queue has too many empty entries
        newlst = []
        for i in range(self.frontIdx, len(self.items)):
            newlst.append(self.items[i])
            
        self.items = newlst
        self.frontIdx = 0
        
    def dequeue(self):
        if self.isEmpty():
            raise RuntimeError("Attempt to dequeue an empty queue")
            # this reaches an amortized complexity of O(1), just similar to what we did in 
            # the list append. If we want to dequeue n element, if we dont compress the queue
            # it would cost O(n).
        if self.frontIdx * 2 > len(self.items):
            # every time we dequeue an element, the frontIdx moved to the next position.
            # so fronIdx is getting bigger, this means the queue is empty in the front.
            # we need to shrink the size if the empty entries reach half of the list.
            self.__compress()
        item = self.items[self.frontIdx]
        self.frontIdx += 1
        return item
    
    def enqueue(self, item):
        self.items.append(item)
        
    def front(self):
        if self.isEmpty():
            raise RuntimeError("Attempt to acess front of empty queue")
        return self.items[self.frontIdx]
    
    def isEmpty(self):
        return self.frontIdx == len(self.items)
```
## Application of S&Q
### Infix Expression Evaluation
#### 1. How to implement our own eval funciton for evaluating infix expression?
An infix expression is an expression where operators appear in between their operands. Like (8 + 9)*2  
Infix evaluator algorithm uses two stacks, an operator stack and an operand stack. The operator stack will hold operators and left parens. The operand stack holds numbers. The algorithm proceeds by scanning the tokens of the infix expression from left to right.  
Each operator has a precedence associated with it:  
/ * >>>  + -  >>> ()  

The stacks are initialized to empty stacks and a left paren is pushed on the operator stack.
1. If the token is an operator then we need to operate on the two stacks with the given operator.
2. If the token is a number then we push it on the number stack.  

After scanning all the input and operating when required, we operate on the stacks one more time with an extra right paren operator.  
#### 2. The operate procedure is a separate function with operator, the operator stack, and operand stack as arguments.  
__a.__ If the given operator is a left paren we push it on the operator stack and return.  
__b.__ Otherwise, if the precedence of the given operator <= the top operator on the operator stack, do as follows:
<ul>
<li> Pop the top operator from the operator stack. Call this the topOp.
<li>if topOp is a +, -，* or / then operate on the number stack by popping the operands, doing the operation, and pushing the result.
<li>if topOp is a left paren, then the given operator should be a right paren. If so,operating is done and return immediately.  
</ul>  

__c.__ When the precedence of the given operator is greater than the precedence of topOp we terminate the loop and push the given operator on the operator stack before returning from operating.



### Radix Sort
This algorithm sorts a sequence of strings lexicographically, as they would appear in a phonebook or dictionary.  
As the strings are read and placed in a queue called __mainQueue__, the algorithm keeps track of the length of the longest string that will be sorted, call it __longest__.  
Radix sort requires all the strings in the mainQueue are of the same length, so those shorter strings should be padded with blanks.
Besides, we also need a helper function that returns a character of a string like __chartAt()__.  
```python 
def charAt(s, i):
    if len(s)-1 < i:
        return " "
    return s[i]
```

This charAt() will make strings in mainQueue look like they are the same length.  
There are also 256 queues created and placed into a list of queues, called queueList. There are 256 different possible ASCII values and one queue is created for each ASCII letter. Queues' indices 0-127, corresponding to the ASCII range.  
1. remove a string from mainQueue, check the last character(starting at __longest-1__), placing the string on a queue that corresponds to the character's ASCII value. using function like __ord("a")__.  
2. starting with the first queue in the queueList, all strings are dequeued from each queue and placed on the main queue.  
3. go back to removing all elements from the main queue again, this time looking at the second to the last letter of each word.
4. The process is repeated until we get to the first letter of each string. When we are done, all strings are on the main queue in sorted order.



