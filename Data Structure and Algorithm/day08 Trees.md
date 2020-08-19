
## 0. Abstract Syntax Trees

A Python program is converted to a tree-like structure called an Abstract Syntax Tree, often abbreviated AST, before it is executed.  
Consider the expression `(5+4)*6+3`. We could define one class for each type of node.  
Calling eval on the root node will recursively call eval on every node in the tree.


```python
class TimesNode:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
    def eval(self):
        return self.left.eval()*self.right.eval()
    
class PlusNode:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
    def eval(self):
        return self.left.eval()+self.right.eval()    
    
class NumNode:
    def __init__(self, num):
        self.num = num
        
    def eval(self):
        return self.num
    
def main():
    x = NumNode(5)
    y = NumNode(4)
    p = PlusNode(x,y)
    t = TimesNode(p, NumNode(6))
    root = PlusNode(t, NumNode(3))
    print(root.eval())
    
if __name__ == "__main__":
    main()
```

An __infix expression__ is an expression written with the binary operators in between their operands. A __postfix expression__ the binary operators are written after their operands.  
eg: infix expression (5 + 4) * 6 + 3 can be written in postfix form as 5 4 + 6 * 3 +.  
Postfix expressions are well-suited for evaluation with a stack. When we come to an operand we push the value on the stack. __When we come to an operator, we pop the operands from the stack, do the operation, and push the result__.  

As another example of a tree traversal, consider writing a method that returns a string representation of an expression. To get a string representing an infix version of the expression, you perform an __inorder traversal__ of the AST. To get a postfix expression you would do a postfix traversal of the tree.


```python
class TimesNode:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
    def eval(self):
        return self.left.eval()*self.right.eval()
    
    def inorder(self):
        return "(" + self.left.inorder() + " * " + self.right.inorder() + ")"
    
class PlusNode:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
    def eval(self):
        return self.left.eval()+self.right.eval()    
    
    def inorder(self):
        return "(" + self.left.inorder() + " + " + self.right.inorder() + ")"
    
class NumNode:
    def __init__(self, num):
        self.num = num
        
    def eval(self):
        return self.num
    
    def inorder(self):
        return str(self.num)
```

To do a postorder traversal of the tree we would write a postorder method that would add each binary operator to the string after postorder traversing the two operands. Because of the way a postorder traversal is written, parentheses are never needed in postfix expressions.  
One other traversal is possible, called a __preorder traversal__. Infix expression (5 + 4) * 6 + 3 is equivalent to  + * + 5 4 6 3 for __prefix expression__. Again, because of the way a prefix expression is written, parentheses are never needed in prefix expressions.  

Abstract syntax trees are built automatically by an interpreter or a compiler.

### The Prefix Expression Grammar
G = (N ,T, P, E) where  
N = {E}   
T = {identifier, number,+, 鈭梷  
P is defined by the set of productions, E is composed of { +E E | 鈭?E E | number}  
a set of non-terminals symbols denoted by N, a set of terminals or tokens called T, and a set, P, of productions. One of the nonterminals is designated the start symbol of the grammar.

The special start symbol E stands for any prefix expression. There are three productions that provide the rules for how prefix expressions can be constructed. The grammar is recursive so every time you see E in the grammar, it can be replaced by another prefix expression.  
This grammar is very easy to convert to a function(__parser__) that given a __queue of tokens__ will build an abstract syntax tree of a prefix expression.


```python
import queue

def E(q):
    if q.isEmpty():
        raise ValueError("Invalid Prefix Expression")
    token = q.dequeue()
    
    if token == "+":
        return PlusNode(E(q), E(q))
    
    if token == "*":
        return TimesNode(E(q), E(q))
    
    return NumNode(float(token))

def main():
#     tokens must be separated by spaces
    x = input("Please enter a prefix expression: ")
    lst = x.split()
    q = queue.Queue()
    
    for token in lst:
        q.enqueue(token)
        
    root = E(q)
    
    print(root.eval())
    print(root.inorder())

if __name__ == "__main__":
    main()
```

## 1. Binary Search Trees
Each node of a binary search tree has up to two children. All values in the left subtree of a node are less than the value at the root of the tree and all values in the right subtree of a node are greater than or equal to the value at the root of the tree.


```python
class BinarySearchTree:
    class __Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
        
        def getVal(self):
            return self.val
        
        def setVal(self, newval):
            self.val = newval
            
        def getLeft(self):
            return self.left
        
        def getRight(self):
            return self.right
#         this set and get operation operate a Node 
        def setLeft(self, newleft):
            self.left = newleft
        
        def setRight(self, newright):
            self.right = newright
            
        def __iter__(self):
#             this does an inorder traversal of the nodes of the tree yielding all the values
#             we can get the values in ascending order            

           if self.left != None:
                for elem in self.left:
                    yield elem
                    
            yield self.val
            
            if self.right != None:
                for elem in self.right:
                    yield elem
    
    def __init__(self):
#         this means this tree class cant be pass with a list to get
#         a tree initially.
        self.root = None
    
    def __iter__(self):
        if self.root != None:
            return self.root.__iter__()
        else:
            return [].__iter__()
        
    def insert(self, val):
        
        def __insert(root, val):
            if root == None:
                return BinarySearchTree.__Node(val)
            
            # think root(5) and we have leftnode(3), then we insert a node(1), what would functions be called?
            if val < root.getVal():
                root.setLeft(__insert(root.getLeft(), val))
            else:
                root.setRight(__insert(root.getRight(), val))
                
            return root
        
        self.root = __insert(self.root, val)
#the insert method will always choose the first element inserted as the root node.
#using the following three trees as example, when you input 1,3,2,5,4 will get the first tree.
#######################################################################################################
#                1               5            3    
#                \              /           / \ 
#                3             3           2   4  
#               / \          /  \         /    \
#             2    5        1    4       1     5
#                 /          \                  
#               4             2      
#######################################################################################################         
        
def main():
    s = input("Enter a list of numbers: ")
    lst = s.split()

    tree = BinarySearchTree()

    for x in lst:
        tree.insert(float(x))

    for x in tree:
        print(x)

if __name__ == "__main__":
    main()
```

### Drawbacks
In the average case, inserting into a binary search tree takes O(log n) time. To insert n items into a binary search tree would take O(n log n) time. However, it takes more __space__ than a list and the quicksort algorithm can sort a list with the same big-Oh complexity. __When the items are already sorted__, both quicksort and binary search trees perform poorly. The complexity of inserting
n items into a binary search tree becomes O(n^2) in the worst case. The tree essentially becomes a linked list.  
### Advantages:
Inserting into a tree can be done in O(log n) time in the average case while inserting into a list would take O(n) time. Deleting from a binary search tree can also be done in O(log n) time in the average case. Looking up a value in a binary search tree can also be done in O(log n) time in the average case.   
A tree datatype can hold information and allow quick insert, delete, and search times.
### Search Spaces
In Sudoku puzzles, the process of guessing, trying to finish the puzzle, and undoing bad guesses is called depth first search. Looking for a goal by making guesses is called a depth first search of a problem space. When a dead end is found we may have to backtrack.   
See the following pseudocode:


```python
def dfs(current, goal):
    if current == goal:
        return [current]
    for next in adjacent(current):
        result = dfs(next, goal)
        if result != None:
            return [current] + result
    return None

# this algorithm returns the path from the current node to the goal node.
```
