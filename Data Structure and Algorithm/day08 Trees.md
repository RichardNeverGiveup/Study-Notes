
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
G = (N ,T, P, E) where  
N = {E}   
T = {identifier, number, +, *  
P is defined by the set of productions, E is composed of { +E E | * E E | number}  



```python

```




