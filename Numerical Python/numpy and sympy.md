

```python
import numpy as np
```

## 0. Numpy


```python
# np.logspace(0,2,5)
# np.linspace(0,10,11)
x = np.array([-1,0,1])
y = np.array([-2,0,2])
X, Y = np.meshgrid(x, y)

f = lambda m, n: n + 10 * m
A = np.fromfunction(f, (6, 6), dtype=int)
# Unlike arrays created by using slices, the arrays returned using fancy indexing and
# Boolean-valued indexing are not views but rather new independent arrays.
#****************************************************
A = np.arange(10)
indices = [2, 4, 6]
B = A[indices]
B[0] = -1
A  # not changed.
A[indices] = -1  # this changes A
A
#************************************************
A = np.arange(10)
B = A[A > 5]
B[0] = -1
A  # not changed.
A[A > 5] = -1    # this changes A
A
#*********************************************
# np.ndarray.flatten  create a copy
# np.ndarray.ravel    create a view
# np.squeeze        Removes axes with length 1
# np.expand_dims    Add a new axis (dimension) of length 1 to an array
# np.newaxis  
data = np.arange(0, 5)
column = data[:, np.newaxis]   # np.expand_dims(data, axis=1) is equivalent
column
row = data[np.newaxis, :]      # np.expand_dims(data, axis=0) is equivalent
row
np.expand_dims(data, axis=1)
np.expand_dims(data, axis=0)
# ********************************************
def heaviside(x):
    return 1 if x > 0 else 0
heaviside = np.vectorize(heaviside)
x = np.linspace(-5, 5, 11)
heaviside(x)
# Although the function returned by np.vectorize works with arrays, it will be relatively slow
# since the original function must be called for each element in the array


# define a function describing a pulse of a given height, width, and position.
def pulse(x, position, height, width):
    return height * (x >= position) * (x <= (position + width))
x = np.linspace(-5, 5, 11)
pulse(x, position=-2, height=1, width=5)
pulse(x, position=1, height=1, width=5)

def pulse(x, position, height, width):
    return height * np.logical_and(x >= position, x <= (position + width))
x = np.linspace(-4, 4, 9)
np.where(x < 0, x**2, x**3)

np.select([x < -1, x < 2, x >= 2], [x**2 , x**3 , x**4])
# output: array( [ 16., 9., 4., -1., 0., 1., 16., 81., 256.])
# for elements in x which is < -1, use x**2, for those x < 2, apply x**3


np.choose([0, 0, 0, 1, 1, 1, 2, 2, 2], [x**2, x**3, x**4])
# output: array( [ 16., 9., 4., -1., 0., 1., 16., 81., 256.])
# [0, 0, 0, 1, 1, 1, 2, 2, 2] means the first three elements use the zero-index function(x**2), the fourth to sixth elements
# use one-index function(x**3)....


# Ap = B.dot(A.dot(np.linalg.inv(B))) may look complicated, so we may want to use np.matrix 
# to make the expressions looks more concise, but it also has some drawbacks.

# Ap = B * A * B.I  looks much more concise, But it is not immediately clear if A * B 
# denotes elementwise or matrix multiplication, because it depends on the type of A and B.

# when we use np.matrix type, it might be a good idea to explicitly cast arrays to matrices 
# before the computation and explicitly cast the result back to the ndarray type.

# A = np.asmatrix(A)
# B = np.asmatrix(B)
# Ap = B * A * B.I
# Ap = np.asarray(Ap)
```

## 1. Symbolic Computing


```python
import sympy
sympy.init_printing()  # configures sympy printing display nicely
from sympy import I, pi, oo
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
