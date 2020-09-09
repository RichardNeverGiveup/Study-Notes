## 1. Equation Solving


```python
# Because linear systems are readily solvable, they are also an important tool for local approximations of nonlinear systems.
# By considering small variations from an expansion point, a nonlinear system can often be approximated 
# by a linear system in the local vicinity of the expansion point.

# For tackling nonlinear problems, we will use the root-finding functions in the optimize module of SciPy.
# use the scipy.linalg module, for solving linear systems of equations.

from scipy import linalg as la
from scipy import optimize
import sympy
sympy.init_printing()
import numpy as np
import matplotlib.pyplot as plt
```


```python
# for Linear Equation Systems, let's focus on the square systems first.
# for Ax=b to have a unique solution, the matrix A must be nonsingular. then x = (inverse of A)*b
# n, rank(A) < n, or detA = 0, then Ax = b can either have no solution or infinitely many solutions, depending on b.

# When A has full rank, the solution is guaranteed to exist. 
# However, it may or may not be possible to accurately compute the solution.
# The condition number of the matrix, cond(A), gives a measure of how well or poorly conditioned a linear equation system is.
# If the conditioning number is close to 1, if the system is said to be well conditioned,
# if the condition number is large, the system is said to be ill-conditioned. 
# The solution to an equation system that is ill-conditioned can have large errors.

# Consider a small variation of b, delta-b, gives a corresponding change in the solution, delta-x, given by A(x+dx) = b + db
# how large is the relative change in x compared to the relative change in b? compare ||dx||/||x|| and ||db||/||b||.
# we can deduct ||dx||/||x||  <=  (||inverse of A||*||A||)*||db||/||b||.
# so we define cond(A) = ||inverse of A||*||A||
# When solving a system of linear equations, it is important to check the condition number to estimate the accuracy of the solution.

# SymPy use the Matrix methods rank, condition_number, and norm.
# numpy use np.linalg.matrix_rank, np.linalg.cond, and np.linalg.norm
```


```python
# 2x_1 + 3X_2 = 4
# 5x_1 + 4X_2 = 3        solving these system
A = sympy.Matrix([[2, 3], [5, 4]])
b = sympy.Matrix([4, 3])
A.rank()
_ = A.condition_number()
sympy.N(_)  # sympy.evalf() is the same
A.norm()

A = np.array([[2, 3], [5, 4]])
b = np.array([4, 3])
np.linalg.matrix_rank(A)
np.linalg.cond(A)
np.linalg.norm(A)
```


```python
A = sympy.Matrix([[2, 3], [5, 4]])
b = sympy.Matrix([4, 3])
L, U, _ = A.LUdecomposition()
L
U
L*U
x = A.solve(b); x
```


```python
A = np.array([[2, 3], [5, 4]])
b = np.array([4, 3])
P, L, U = la.lu(A)
L
P.dot(L.dot(U))
la.solve(A, b)
```


```python
p = sympy.symbols("p", positive=True)
A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]])
b = sympy.Matrix([1, 2])
x = A.solve(b)
x
```


```python
p = sympy.symbols("p", positive=True)
A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]])
b = sympy.Matrix([1, 2])
# Solve symbolically
x_sym_sol = A.solve(b)
Acond = A.condition_number().simplify()

# Numerical problem specification
AA = lambda p: np.array([[1, np.sqrt(p)], [1, 1/np.sqrt(p)]])  # turn varible p into an array   
bb = np.array([1, 2])
x_num_sol = lambda p: np.linalg.solve(AA(p), bb)    # solution is an array

# Graph the difference between the symbolic (exact) and numerical results.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

p_vec = np.linspace(0.9, 1.1, 200)
for n in range(2):
#******************************these two lines need to memorize*************************************************************
#*************************they showed the difference between sympy and numpy************************************************
    x_sym = np.array([x_sym_sol[n].subs(p, pp).evalf() for pp in p_vec])  # sympy syntax
    x_num = np.array([x_num_sol(pp)[n] for pp in p_vec])
#***************************************************************************************************************************
    axes[0].plot(p_vec, (x_num - x_sym)/x_sym, 'k')
axes[0].set_title("Error in solution\n(numerical - symbolic)/symbolic")
axes[0].set_xlabel(r'$p$', fontsize=18)

axes[1].plot(p_vec, [Acond.subs(p, pp).evalf() for pp in p_vec])
axes[1].set_title("Condition number")
axes[1].set_xlabel(r'$p$', fontsize=18)
```


```python
# now let's turn to rectangular systems
x_vars = sympy.symbols("x_1, x_2, x_3")
A = sympy.Matrix([[1, 2, 3], [4, 5, 6]])
x = sympy.Matrix(x_vars)
b = sympy.Matrix([7, 8])
sympy.solve(A*x - b, x_vars)
```


```python
# overdetermined means system has more equations than unknown variables, m > n.
# in general there is no exact solution to such a system. 
# However, it is often interesting to find an approximate solution to an overdetermined system.

# Eg: data fitting, say we have a model where a variable y is a quadratic polynomial in the variable x, 
# so that y = A+Bx+Cx^2, and that we would like to fit this model to experimental data.
# y is nonlinear in x, but y is linear in the three unknown coefficients A, B, and C.
# we can collect m pairs (x,y) data and transform them into (1, x, x^2) and y, then we can get A, B, C.
# if the data is noisy and if we were to use more than three data points, we should be able to get a more accurate
# estimate of the model parameters.

# r is the residual vector, r = b - Ax. then we can try to minimize the sum of square errors(r^2).

# In SymPy we can solve for the least square solution of an overdetermined system using 
# the solve_least_squares method, and for numerical problems, we can use the SciPy function la.lstsq.
x = np.linspace(-1, 1, 100)
a, b, c = 1, 2, 3
y_exact = a + b * x + c * x**2

# simulate noisy data
m = 100
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + np.random.randn(m)

# fit the data to the model using linear least square
A = np.vstack([X**0, X**1, X**2]) # see np.vander for alternative    # 3*m matrix
sol, r, rank, sv = la.lstsq(A.T, Y)   # A.T is a m*3 matrix

y_fit = sol[0] + sol[1] * x + sol[2] * x**2
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2)
```


```python
# we can fit the same data used in the previous example to linear model and to a higher-order polynomial model (up to order 15).
# The former case corresponds to underfitting, and the latter case corresponds to overfitting.

# fit the data to the model using linear least square:
# 1st order polynomial
#**************************************************************************************
A = np.vstack([X**n for n in range(2)])   # here controls the order of x
#**************************************************************************************
sol, r, rank, sv = la.lstsq(A.T, Y)
y_fit1 = sum([s * x**n for n, s in enumerate(sol)])  # A*x**0 + B*x**1 + .....

# 15th order polynomial
#**************************************************************************************
A = np.vstack([X**n for n in range(16)])    # here controls the order of x
#**************************************************************************************
sol, r, rank, sv = la.lstsq(A.T, Y)
y_fit15 = sum([s * x**n for n, s in enumerate(sol)])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
ax.plot(x, y_fit1, 'b', lw=2, label='Least square fit [1st order]')
ax.plot(x, y_fit15, 'm', lw=2, label='Least square fit [15th order]')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2)
```


```python
# eigenvalue equation Ax = cx, where A is a N 脳 N square matrix
# Here x is an eigenvector and c is an eigenvalue of the matrix A.

# (A 鈭?Ic)x = 0 and there to exist a nontrivial solution, x 鈮?0, 
# the matrix A 鈭?Ic must be singular, and its determinant must be zero.
eps, delta = sympy.symbols("epsilon, Delta")
H = sympy.Matrix([[eps, delta], [delta, -eps]])
H
```


```python
H.eigenvals()
```


```python
H.eigenvects()
```


```python
(eval1, _, evec1), (eval2, _, evec2) = H.eigenvects()
sympy.simplify(evec1[0].T * evec2[0])
```


```python
# for larger systems we must use numerical approach. la.eigvals and la.eig functions in the SciPy linear algebra package.
A = np.array([[1, 3, 5], [3, 5, 3], [5, 3, 9]])
evals, evecs = la.eig(A)
la.eigvalsh(A)   # la.eigvalsh and la.eigh guarantee the eigenvalues returned with real values.
```


```python
# Univariate Equations
# Unlike for linear systems, there are no general methods for determining if a nonlinear equation has a solution,
# or multiple solutions, or if a given solution is unique. This can be understood intuitively from the fact 
# that graphs of nonlinear functions correspond to curves that can intersect x = 0 in an arbitrary number of ways.
x, a, b, c = sympy.symbols("x, a, b, c")
sympy.solve(a + b*x + c*x**2, x)
```


```python
sympy.solve(a * sympy.cos(x) - b * sympy.sin(x), x)
```


```python
# However, in general nonlinear equations are typically not solvable analytically. So we often use numerical root finding.
# Two standard methods are the bisection method and the Newton method.

# The bisection method requires a starting interval [a, b] such that f(a) and f(b) havedifferent signs.
# In each iteration, the function is evaluated in the middle point m between a and b. 
# sign of the function is different at a and m, and then the new interval [a, b = m] is chosen for the next iteration.
# Otherwise the interval [a = m, b] is chosen.

# define a function, desired tolerance and starting interval [a, b]
f = lambda x: np.exp(x) - 2
tol = 0.1
a, b = -2, 2
x = np.linspace(-2.1, 2.1, 1000)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax.plot(x, f(x), lw=1.5)
ax.axhline(0, ls=':', color='k')
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$f(x)$', fontsize=18)

fa, fb = f(a), f(b)
ax.plot(a, fa, 'ko')
ax.plot(b, fb, 'ko')
ax.text(a, fa + 0.5, r"$a$", ha='center', fontsize=18)
ax.text(b, fb + 0.5, r"$b$", ha='center', fontsize=18)

n = 1
while b - a > tol:
    m = a + (b - a)/2
    fm = f(m)
    ax.plot(m, fm, 'ko')
    ax.text(m, fm - 0.5, r"$m_%d$" % n, ha='center')
    n += 1
    
    if np.sign(fa) == np.sign(fm):
        a, fa = m, fm
    else:
        b, fb = m, fm
    
ax.plot(m, fm, 'r*', markersize=10)
ax.annotate("Root approximately at %.3f" % m,
            fontsize=14, family="serif",
            xy=(a, fm), xycoords='data',
            xytext=(-150, +50), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
            connectionstyle="arc3, rad=-.5"))    
```


```python
# Newton's method approximates the f(x) with its first-order Taylor expansion f(x+dx) = f(x)+dx f'(x), 
# which is a linear function whose root is easily found to be x 鈥?f(x)/f'(x).
# By iterating this scheme, X_(k+1) = X_k 鈭?f(X_k)/f'(X_k), we may approach the root of the function.
# A potential problem with this method is that it fails if f'(x) is zero at some point.

# define a function, desired tolerance and starting point xk
tol = 0.01
xk = 2

s_x = sympy.symbols("x")
s_f = sympy.exp(s_x) - 2
#*************************************************************************************************************
# these two lines need to memorize
f = lambda x: sympy.lambdify(s_x, s_f, 'numpy')(x)
fp = lambda x: sympy.lambdify(s_x, sympy.diff(s_f, s_x), 'numpy')(x)
#*************************************************************************************************************
x = np.linspace(-1, 2.1, 1000)

# setup a graph for visualizing the root finding steps
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(x, f(x))
ax.axhline(0, ls=':', color='k')

n = 0
while f(xk) > tol:       # we need f(xk) = 0 to get the root
    xk_new = xk - f(xk) / fp(xk)      # the root of first-order Taylor expansion, if f(xk_new) > tol need to keep iterate.
    ax.plot([xk, xk], [0, f(xk)], color='k', ls=':')
    ax.plot(xk, f(xk), 'ko')
    ax.text(xk, -.5, r'$x_%d$' % n, ha='center')  # xk position and -0.5 lower than y=0
    ax.plot([xk, xk_new], [f(xk), 0], 'k-')

    xk = xk_new
    n += 1

ax.plot(xk, f(xk), 'r*', markersize=15)
ax.annotate("Root approximately at %.3f" % xk,
            fontsize=14, family="serif",
            xy=(xk, f(xk)), xycoords='data',
            xytext=(-150, +50), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
            connectionstyle="arc3, rad=-.5"))
ax.set_title("Newtown's method")
ax.set_xticks([-1, 0, 1, 2])
```


```python
# SciPy optimize module provides multiple functions for numerical root finding.eg: optimize.bisect and optimize.newton functions.
optimize.bisect(lambda x: np.exp(x) - 2, -2, 2)
# Newton's method takes a function as the first argument and an initial guess.
x_root_guess = 2
f = lambda x: np.exp(x) 鈥?2
fprime = lambda x: np.exp(x)
optimize.newton(f, x_root_guess)
optimize.newton(f, x_root_guess, fprime=fprime)
# optimize.brentq and optimize.brenth functions, which are variants of the bisection method.
# optimize.brentq function is generally considered the preferred all-around root-finding function in SciPy.
optimize.brentq(lambda x: np.exp(x) - 2, -2, 2)
optimize.brenth(lambda x: np.exp(x) - 2, -2, 2)
```


```python
# Systems of Nonlinear Equations

# we cannot in general write a system of nonlinear equations as a matrix-vector multiplication. 
# Instead we represent a system of multivariate nonlinear equations as a vector-valued function. f:R^N -> R^N
# It takes an N-dimensional vector and maps it to another N-dimensional vector.

# bisection method cannot be directly generalized to a multivariate equation system.
# Newton's method can be used for multivariate problems.
# the Jacobian matrix of the function f(x) is used to do iterations.

# Like the secant variants of the Newton method for univariate equation systems, there are also
# variants of the multivariate method that avoid computing the Jacobian.
# Broyden's method is a popular example of this type of secant updating method for multivariate equation systems.
#**********************************************************************************************************************************
# optimize.fsolve provides an implementation of a Newton-like method, where optionally the Jacobian can be specified, if available.
# The first argument is a Python function that represents the equation to be solved, and it should
# take a NumPy array as the first argument and return an array of the same shape.
#**********************************************************************************************************************************


# y - x^3 - 2x^2 + 1 = 0
# y + x^2 - 1 = 0
# solve the above equation system, f([x1, x2]) = [x2 鈭?x1^3 鈭?2x1^2+1, x2+x1^2 鈭?1].

def f(x):
    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]
optimize.fsolve(f, [1, 1])

x, y = sympy.symbols("x, y")
f_mat = sympy.Matrix([y - x**3 -2*x**2 + 1, y + x**2 - 1])
f_mat.jacobian(sympy.Matrix([x, y]))
```


```python
def f_jacobian(x):
    return [[-3*x[0]**2-4*x[0], 1], [2*x[0], 1]]
optimize.fsolve(f, [1, 1], fprime=f_jacobian)
```


```python
def f(x):
    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]

x = np.linspace(-3, 2, 5000)
y1 = x**3 + 2 * x**2 -1
y2 = -x**2 + 1

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, y1, 'b', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, 'g', lw=1.5, label=r'$y = -x^2 + 1$')

x_guesses = [[-2, 2], [1, -1], [-2, -5]]
for x_guess in x_guesses:
    sol = optimize.fsolve(f, x_guess)
    ax.plot(sol[0], sol[1], 'r*', markersize=15)
    ax.plot(x_guess[0], x_guess[1], 'ko')
    ax.annotate( "", xy=(sol[0], sol[1]), xytext=(x_guess[0], x_guess[1]),
                arrowprops=dict(arrowstyle="->", linewidth=2.5))
ax.legend(loc=0)
ax.set_xlabel(r'$x$', fontsize=18)
```


```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y1, 'k', lw=1.5)
ax.plot(x, y2, 'k', lw=1.5)
sol1 = optimize.fsolve(f, [-2, 2])
sol2 = optimize.fsolve(f, [ 1, -1])
sol3 = optimize.fsolve(f, [-2, -5])
sols = [sol1, sol2, sol3]
colors = ['r', 'b', 'g']
for idx, s in enumerate(sols):
    ax.plot(s[0], s[1], colors[idx]+'*', markersize=15)
for m in np.linspace(-4, 3, 80):
    for n in np.linspace(-15, 15, 40):
        x_guess = [m, n]
        sol = optimize.fsolve(f, x_guess)
        idx = (abs(sols - sol)**2).sum(axis=1).argmin()   # count which solution is the closest.
        ax.plot(x_guess[0], x_guess[1], colors[idx]+'.')
ax.set_xlabel(r'$x$', fontsize=18)
```

## Conclusion:
Linear equation systems can always be systematically solved, but nonlinear may not. We could apply linear approximations to solve some of the nonlinear equation systems.
### Linear equation systems:
1. Square systems: watch out the condion number(due to floating point erro) if we use numerical solutions in scipy and numpy 
2. Rectangular systems: underdetermined and overdetermined, some tricks about data fitting, 
3. Eigenvalue problems: N * N matrix 

### Nonlinear equation:
1. Univariate equations: newtown method and bisection method
2. Systems of nonlinear equations: Jacobian matrix
