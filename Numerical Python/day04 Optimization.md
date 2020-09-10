
## Optimization


```python
# use SciPy's optimization module optimize for nonlinear optimization problems, 
# the convex optimization library cvxopt for linear optimization problems with linear constraints.
import cvxopt
from scipy import optimize
import matplotlib.pyplot as plt
import sympy
import numpy as np
sympy.init_printing()
```

If the objective function and the constraints all are linear, the problem is a linear optimization problem. If either the objective function or the constraints are nonlinear, it is a nonlinear optimization problem, or a nonlinear programming problem.  
A general nonlinear problem can have both local and global minima, which turns out to make it very difficult to find the global minima. An important subclass of nonlinear problems that can be solved efficiently is convex problems, which is directly related to the absence of strictly local minima and the existence of a unique global minimum.  
Whether the objective function f(x) and the constraints g(x) and h(x) are continuous and smooth are properties that have very important implications for the methods and techniques that can be used to solve an optimization problem. If the function itself is
not known exactly, but contains noise due to measurements or for other reasons, many of the methods discussed in the following may not be suitable.  
If the second-order derivative is positive, or the Hessian positive definite, when evaluated at stationary point, then that point is a local minimum..

### Univariate optimization
In optimize module, the brent function is a hybrid method, and it is generally the preferred method for optimization of univariate functions with SciPy. This method is a variant of the golden section search method that uses inverse parabolic interpolation to obtain faster convergence.  
Instead of calling the optimize.golden and optimize.brent functions directly, we can use the unified interface function optimize.minimize_scalar, which dispatches to the optimize.golden and optimize.brent functions depending on the keyword argument passing in.  
Optimize.fminbound performs optimization on a bounded interval, which corresponds to an optimization problem with inequality constraints that limit the domain of objective function.  
If we wanna:  
Minimize the area of a cylinder with unit volume.


```python
r, h = sympy.symbols("r, h")
Area = 2 * sympy.pi * r**2 + 2 * sympy.pi * r * h
Volume = sympy.pi * r**2 * h
h_r = sympy.solve(Volume - 1)[0]
Area_r = Area.subs(h_r)
rsol = sympy.solve(Area_r.diff(r))[0]
Area_r.diff(r, 2).subs(r, rsol)
Area_r.subs(r, rsol)
```


```python
def f(r):
    return 2 * np.pi * r**2 + 2 / r
r_min = optimize.brent(f, brack=(0.1, 4))
f(r_min)

optimize.minimize_scalar(f, bracket=(0.1, 4))

# When possible, it is a good idea to visualize the objective function before attempting a numerical optimization, 
# because it can help in identifying a suitable initial interval or a starting point for the numerical optimization routine.
```

### Unconstrained Multivariate Optimization
Analytical approach of solving the nonlinear equations for roots of the gradient is rarely feasible in the multivariate case, and the bracketing scheme used in the golden search method is also not directly applicable. Instead we must resort to techniques that start at some point in the coordinate space and use different strategies to move toward a better approximation of the minimum point.
  
  
The most basic approach of this type is to consider the gradient of the objective function f(x) at a given point x. In general, the negative of the gradient always points in the direction in which the function f(x) decreases the most. Move along this direction for
some distance k and then iterate this scheme at the new point. This method is known as the steepest descent method, k is a free parameter called line search parameter. The convergence of this method can be quite slow, since it would produce zigzag approach to the minimum.  

Newton's method for multivariate optimization is a modification of the steepest descent method that can improve convergence.
As in the univariate case, Newton's method can be viewed as a local quadratic approximation of the function, which when minimized gives an iteration scheme. In the multivariate case, the iteration formula compared to the steepest descent method, the gradient has been replaced with the gradient multiplied from the left with the inverse of Hessian matrix for the function. This alters both the direction and the length of the step, this may not converge if started too far from a minimum, but when close to a minimum, it converges quickly. As usual there is a trade-off between convergence rate and stability.
  
In SciPy, Newton鈥檚 method is implemented in the function optimize.fmin_ncg. It takes the following arguments: a Python function for the objective function, a starting point, a Python function for evaluating the gradient, and (optionally) a Python function for evaluating the Hessian.


```python
x1, x2 = sympy.symbols("x_1, x_2")
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2
fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]
sympy.Matrix(fprime_sym)
fhess_sym = [[f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] for x2_ in (x1, x2)]
sympy.Matrix(fhess_sym)
f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy')    # this is the tricky part, this f_lmbd takes two seperate arrays as input
fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')
fhess_lmbda = sympy.lambdify((x1, x2), fhess_sym, 'numpy')
##################################################################################################################
def func_XY_to_X_Y(f):
    '''Wrapper for f(X) -> f(X[0], X[1])'''
    return lambda X: np.array(f(X[0], X[1])) # split one array into two arrays
######################################################################################################################
f = func_XY_to_X_Y(f_lmbda)
fprime = func_XY_to_X_Y(fprime_lmbda)
fhess = func_XY_to_X_Y(fhess_lmbda)
x_opt = optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)
x_opt
```


```python
#************************************************************************************
# the normal use of lambdify
x = sympy.symbols('x')
expr = x**2+x*6
f = sympy.lambdify(x, expr, 'numpy')
a = np.array([1, 2, 3, 4, 5])
f(a)   
#************************************************************************************
```


```python
#*******************************************************************************************************
x1, x2 = sympy.symbols("x_1, x_2")
f_sym = x1 + x2
g = sympy.lambdify((x1,x2), f_sym, 'numpy')
# g((2,3))  # this way g wont work, need to use g(2,3)
# n = np.array([(1,2),(2,3),(3,4),(4,5)])   this wont work neither
# g(n)
# so, if we used (x1, x2) in lambdify, we need to use two arrays as input.
# n = np.array([[1,2,3,4,5],[6,7,8,9,10]]) # this wont work, even we packed two arrays into one
# g(n)       
#******************************************************************************************************
def func_XY_to_X_Y(f):
    '''Wrapper for f(X) -> f(X[0], X[1])'''
    return lambda X: np.array(f(X[0], X[1])) # split one array into two arrays

test = func_XY_to_X_Y(g)
# after we used the wraper, now we can use one array as input!
# n = np.array([[all the x1],[all the x2]])
x1x2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
test(n)
```


```python
# this is the visualization
fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, f_lmbda(X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
```

It may not always be possible to provide functions for both the gradient and the Hessian of the objective function, and often it is convenient with a solver that only requires function evaluations. Several methods exist to numerically estimate the gradient or the Hessian. Methods that approximate the Hessian are known as quasi-Newton methods, and there are also alternative iterative methods that completely avoid using the Hessian.  
Two popular methods are the Broyden-Fletcher-Goldfarb-Shanno (BFGS) and the conjugate gradient methods, which are implemented in SciPy as the functions __optimize.fmin_bfgs and optimize.fmin_cg__.  
1. The BFGS method is a quasi-Newton method that can gradually build up numerical estimates of the Hessian, and also the gradient. 
2. The conjugate gradient method is a variant of the steepest descent method and does not use the Hessian, and it can be used with numerical estimates of the gradient obtained from only function evaluations  

Both optimize.fmin_bfgs and optimize.fmin_cg can optionally accept a function for evaluating the gradient, but if not provided, the gradient is estimated from function evaluations.  
The previous problem given, which was solved with the Newton method, can also be solved using the optimize.fmin_bfgs and optimize.fmin_cg, without providing a function for the Hessian:


```python
x_opt = optimize.fmin_bfgs(f, (0, 0), fprime=fprime)
x_opt = optimize.fmin_cg(f, (0, 0), fprime=fprime)

x_opt = optimize.fmin_bfgs(f, (0, 0))
```

The methods for multivariate optimization that we have discussed so far all converge to a local minimum in general. For problems with many local minima, although there is no complete and general solution to this problem, a practical approach is to use a brute force search over a coordinate grid to find a suitable starting point. In SciPy, the function optimize.brute can carry out such a
systematic search.


```python
# We need a wrapper function for reshuffling the parameters of the objective function 
# because of the different conventions of how the coordinated vectors are passed to the function.
def f(X):
    x, y = X
    return (4 * np.sin(np.pi * x) + 6 * np.sin(np.pi * y)) + (x - 1)**2 + (y - 1)**2

x_start = optimize.brute(f, (slice(-3, 5, 0.5), slice(-3, 5, 0.5)), finish=None)
x_start
f(x_start)
# This is now a good starting point for a more sophisticated iterative solver, such as optimize.fmin_bfgs:
x_opt = optimize.fmin_bfgs(f, x_start)
```


```python
# since our f need one X as param, so we need a wrapper to make separate arrays packed into one array.
def func_X_Y_to_XY(f, X, Y):
    '''Wrapper for f(X, Y) -> f([X, Y])'''
    s = np.shape(X)
    return f(np.vstack([X.ravel(), Y.ravel()])).reshape(*s)
fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-3, 5, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 25)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
```

SciPy also provides a unified interface for all multivariate optimization solver with the function __optimize.minimize__, which dispatches out to the solver-specific functions depending on the value of the method keyword argument. Remember, the __univariate__ minimization function that provides a unified interface is __optimize.scalar_minimize__.  
The optimize.minimize function returns an instance of optimize.OptimizeResult. The solution is available via the x attribute of this class.


```python
x_opt = optimize.fmin_bfgs(f, x_start)
result = optimize.minimize(f, x_start, method= 'BFGS')
x_opt = result.x
```

### Nonlinear Least Square Problems
A least square problem can be viewed as an optimization problem with the objective function $g(\beta ) = \sum_{i=0}^{m}{r_{i}}\left ( \beta  \right )^{2}$, where $r(\beta )$ is a vector with the residuals $r_{i}\left ( \beta \right ) = y_{i} - f\left ( x_{i},\beta  \right )$ for a set of m observations $(x_{i},y_{i})$. Here $\beta$ is a vector with unknown parameters that specifies the function f.  
If this problem is nonlinear in the parameters $\beta$, it is known as a nonlinear least square problem, and since it is nonlinear, it cannot be solved with the normal linear algebra techniques(like that data fitting example).  
Instead, we can use the multivariate optimization techniques such as Newton's method or a quasi-Newton method. However, this nonlinear least square optimization problem has a specific structure, and several methods that are tailored for it. One example is the Levenberg-Marquardt method, which is based on the idea of successive linearizations of the problem in each iteration. __optimize.leastsq__ provides a nonlinear least square solver that uses the this method.  

Consider a nonlinear model on the form $f(x,\beta) = \beta_{0}+\beta_{1}*exp(-\beta_{2}*x^{2})$


```python
beta = (0.25, 0.75, 0.5)
def f(x, b0, b1, b2):
    return b0 + b1 * np.exp(-b2 * x**2)

xdata = np.linspace(0, 5, 50)
y = f(xdata, *beta)
ydata = y + 0.05 * np.random.randn(len(xdata))

def g(beta):
    return ydata - f(xdata, *beta)
beta_start = (1, 1, 1)
beta_opt, beta_cov = optimize.leastsq(g, beta_start)
beta_opt
```


```python
fig, ax = plt.subplots()
ax.scatter(xdata, ydata, label='samples')
ax.plot(xdata, y, 'r', lw=2, label='true model')
ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2, label='fitted model')
ax.set_xlim(0, 5)
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
ax.legend()
```


```python
# The SciPy optimize module also provides an alternative interface to nonlinear least square fitting, 
# through the function optimize.curve_fit. This is a convenience wrapper around optimize.leastsq, 
# which eliminates the need to explicitly define the residual function for the least square problem.
beta_opt, beta_cov = optimize.curve_fit(f, xdata, ydata)
```

### Constrained Optimization
If it only restricts the range of the coordinate without dependencies on the other variables. we can use L-BFGS-B method in SciPy, this solver is available through the function __optimize.fmin_l_bgfs_b__ or via __optimize.minimize__ with the method argument set to __'L-BFGS-B'__. To define the coordinate boundaries, the bound keyword argument must be used, and its value should be a list of tuples that contain the minimum and maximum value of each constrained variable. If the minimum or maximum value is set to None, it is interpreted as an unbounded.


```python
def func_X_Y_to_XY(f, X, Y):
    '''Wrapper for f(X, Y) -> f([X, Y])'''
    s = np.shape(X)
    return f(np.vstack([X.ravel(), Y.ravel()])).reshape(*s)

def f(X):
    x, y = X
    return (x - 1)**2 + (y - 1)**2
x_opt = optimize.minimize(f, [1, 1], method='BFGS').x
bnd_x1, bnd_x2 = (2, 3), (0, 2)
x_cons_opt = optimize.minimize(f, [1, 1], method='L-BFGS-B', bounds=[bnd_x1, bnd_x2]).x

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)  # wraper is a tricky part
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]),
                            bnd_x1[1] - bnd_x1[0], bnd_x2[1] -
                            bnd_x2[0], facecolor="grey")
ax.add_patch(bound_rect)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
```

Constraints that are defined by equalities or inequalities that include more than one variable are somewhat more complicated to deal with. However, there are general techniques also for this type of problems. For example, using the Lagrange multipliers, it is possible to convert a constrained optimization problem to an unconstrained problem by introducing additional variables.  



```python

```


```python

```


```python

```
