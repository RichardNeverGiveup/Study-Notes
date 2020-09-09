

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
import numpy



# SymPy represents mathematical symbols as Python objects.
x = sympy.Symbol("x")
y = sympy.Symbol("y", real=True)
y.is_real
x.is_real is None
sympy.Symbol("z", imaginary=True).is_real
# the keyword arguments in Symbol Class constructor has their corresponding attributes
# real, imaginary>>>is_real, is_imaginary
# positive, negative>>>is_positive, is_negative
x = sympy.Symbol("x")
sympy.sqrt(x ** 2)
y = sympy.Symbol("y", positive=True)
sympy.sqrt(y ** 2)

n1 = sympy.Symbol("n")
n2 = sympy.Symbol("n", integer=True)
n3 = sympy.Symbol("n", odd=True)
sympy.cos(n1 * pi)
sympy.cos(n2 * pi)
sympy.cos(n3 * pi)

a, b, c = sympy.symbols("a, b, c", negative=True)
# we cannot directly use the built-in Python objects for int, float and so on. Because they do not have Symbol instance attributes
# like is_real, so we use sympy.Integer and sympy.Float instead.
# SymPy automatically promotes Python numbers to instances of these classes when they occur in SymPy expressions.
i, f = sympy.sympify(19), sympy.sympify(2.3)
type(i), type(f)

# the form is_Name indicate if the object is of type Name
# the form is_name indicate if the object is known to satisfy the condition name
n = sympy.Symbol("n", integer=True)
n.is_integer, n.is_Integer, n.is_positive, n.is_Symbol
i = sympy.Integer(19)
i.is_integer, i.is_Integer, i.is_positive, i.is_Symbol

sympy.Float(0.3, 25)
sympy.Float('0.3', 25)  # it is necessary to initialize it from a string '0.3' rather than the Python float 0.3

x, y, z = sympy.symbols("x, y, z")
f = sympy.Function("f")
type(f)
f(x)

g = sympy.Function("g")(x, y, z)
g.free_symbols

sympy.Lambda(x, x**2)

expr = 1 + 2 * x**2 + 3 * x**3
expr
# SymPy expressions should be treated as immutable objects and return new expression trees rather than modify expressions in place.
expr = 2 * (x**2 - x) - x * (x + 1)
sympy.simplify(expr)
expr.simplify()


# expand
expr = (x + 1) * (x + 2)
sympy.expand(expr)
sympy.sin(x + y).expand(trig=True)

a, b = sympy.symbols("a, b", positive=True)
sympy.log(a * b).expand(log=True)
sympy.exp(I*a + b).expand(complex=True)
sympy.expand((a * b)**x, power_base=True)
sympy.exp((a-b)*x).expand(power_exp=True)

# factor, collect, combine
sympy.factor(x**2 - 1)
sympy.factor(x * sympy.cos(y) + sympy.sin(z) * x)
sympy.logcombine(sympy.log(a) - sympy.log(b))
expr = x + y + x * y * z
expr.collect(x)

# apart, together, cancel
sympy.apart(1/(x**2 + 3*x + 2), x)
sympy.together(1 / (y * x + y) + 1 / (1+x))
sympy.cancel(y / (y * x + y))

# substitutions
sympy.sin(x * sympy.exp(x)).subs(x, y)
sympy.sin(x * z).subs({z: sympy.exp(y), x: y, sympy.sin: sympy.cos})
expr = x * y + z**2 *x
values = {x: 1.25, y: 0.4, z: 3.2}
expr.subs(values)

# numerical evaluation
sympy.N(pi, 50)
(x + 1/pi).evalf(10)  # 10 digits after 0

expr = sympy.sin(pi * x * sympy.exp(x))
[expr.subs(x, xx).evalf(3) for xx in range(0, 10)] # this is slow
expr_func = sympy.lambdify(x, expr)   # this way is faster
expr_func(1.0)
expr_func = sympy.lambdify(x, expr, 'numpy')  # vectorized function
xvalues = np.arange(0, 10)
expr_func(xvalues)

f = sympy.Function('f')(x)
f.diff(x)     # sympy.diff(f, x) is equivalent

sympy.diff(f, x, x)
sympy.diff(f, x, 3)
g = sympy.Function('g')(x, y)
g.diff(x, y)    # sympy.diff(g, x, y) is equivalent
g.diff(x, 3, y, 2)  # equivalent to sympy.diff(g, x, x, x, y, y)
# calling sympy.diff on an expression will directly give you results
# if we wanna symbolically represent the derivative of a definite expression, we can use sympy.Derivative()
d = sympy.Derivative(sympy.exp(sympy.cos(x)), x)
d
d.doit()  # this will give you the result.

a, b, x, y = sympy.symbols("a, b, x, y")
f = sympy.Function("f")(x)
sympy.integrate(f)
sympy.integrate(f, (x, a, b))
sympy.integrate(sympy.sin(x), (x, a, b))
sympy.integrate(sympy.exp(-x**2), (x, 0, oo))

expr = (x + y)**2
sympy.integrate(expr, x, y)
sympy.integrate(expr, (x, 0, 1), (y, 0, 1))

sympy.series(f, x)
x0 = sympy.Symbol("{x_0}")
f.series(x, x0, n=2)
f.series(x, x0, n=2).removeO()

sympy.cos(x).series()
sympy.sin(x).series()
sympy.exp(x).series()
(1/(1+x)).series()

sympy.limit(sympy.sin(x) / x, x, 0)

f = sympy.Function('f')
x, h = sympy.symbols("x, h")
diff_limit = (f(x + h) - f(x))/h
sympy.limit(diff_limit.subs(f, sympy.cos), h, 0)
sympy.limit(diff_limit.subs(f, sympy.sin), h, 0)

# main use of limits is to find the asymptotic behavior
expr = (x**2 - 3*x) / (2*x - 2)
p = sympy.limit(expr/x, x, sympy.oo)
q = sympy.limit(expr - p*x, x, sympy.oo)
p, q
sympy.Lambda(x, p * x + q)

n = sympy.symbols("n", integer=True)
x = sympy.Sum(1/(n**2), (n, 1, oo))
x
x.doit()

x = sympy.Product(n, (n, 1, 7))
x
x.doit()

# Equations
x = sympy.Symbol("x")
sympy.solve(x**2 + 2*x - 3)
a, b, c = sympy.symbols("a, b, c")
sympy.solve(a * x**2 + b * x + c, x)

eq1 = x**2 - y
eq2 = y**2 - x
sols = sympy.solve([eq1, eq2], [x, y], dict=True)
sols
[eq1.subs(sol).simplify() == 0 and eq2.subs(sol).simplify() == 0 for sol in sols] # solutions validation

# Linear algebra
sympy.Matrix([1, 2]) # column vector
sympy.Matrix([[1, 2]])  # row vector
sympy.Matrix([[1, 2], [3, 4]])
sympy.Matrix(3, 4, lambda m, n: 10 * m + n)

p, q = sympy.symbols("p, q")
M = sympy.Matrix([[1, p], [q, 1]])
b = sympy.Matrix(sympy.symbols("b_1, b_2"))
x = M.LUsolve(b)
x = M.inv() * b
# However, computing the inverse of a matrix is more difficult than performing the
# LU factorization, so if solving the equation Mx = b is the objective, as it was here, then
# using LU factorization is more efficient.
```



