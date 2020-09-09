
## 0. Plotting and Visualization


```python
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import sympy
```


```python
x = np.linspace(-5, 2, 100)
y1 = x**3 + 5*x**2 + 10
y2 = 3*x**2 + 10*x
y3 = 6*x + 10
fig, ax = plt.subplots()
ax.plot(x, y1, color="blue", label="y(x)")
ax.plot(x, y2, color="red", label="y'(x)")
ax.plot(x, y3, color="green", label="y鈥?x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
```


```python
# the graphical user interface using a variety of different widget toolkits (e.g., Qt, GTK, wxWidgets, and Cocoa for MacOS X)
# that are suitable for different platforms.
# Which backend to use can be selected in that Matplotlib resource file, or using the function mpl.use, 
# which must be called right after importing matplotlib, before importing the matplotlib.pyplot module.

import matplotlib as mpl
mpl.use('qt4agg')
import matplotlib.pyplot as plt
```


```python
# A Figure object can be created using the function plt.figure. Once a Figure is created, we can use the add_axes method 
# to create a new Axes instance.
fig = plt.figure(figsize=(8, 2.5), facecolor="#f1f1f1")
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes((left, bottom, width, height), facecolor="#e1e1e1")
x = np.linspace(-2, 2, 1000)
y1 = np.cos(40 * x)
y2 = np.exp(-x**2)
ax.plot(x, y1 * y2)
ax.plot(x, y2, 'g')
ax.plot(x, -y2, 'g')
ax.set_xlabel("x")
ax.set_ylabel("y")
# fig.savefig("graph.png", dpi=100, facecolor="#f1f1f1")
```


```python
# The Axes object provides the coordinate system and contains the axis objects that determine where the axis labels and the
# axis ticks are placed. The functions for drawing different types of plots are also methods of this Axes class.
# fig, axes = plt.subplots(nrows=3, ncols=2)

# The plt.subplots function also takes two special keyword arguments fig_kw and subplot_kw, 
# which are dictionaries with keyword arguments that are used when creating the Figure and Axes instances.
x = np.linspace(-5, 5, 5)
y = np.ones_like(x)
def axes_settings(fig, ax, title, ymax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, ymax+1)
    ax.set_title(title)
fig, axes = plt.subplots(1, 4, figsize=(16,3))
linewidths = [0.5, 1.0, 2.0, 4.0]
for n, linewidth in enumerate(linewidths):
    axes[0].plot(x, y + n, color="blue", linewidth=linewidth)
    axes_settings(fig, axes[0], "linewidth", len(linewidths))

linestyles = ['-', '-.', ':']
for n, linestyle in enumerate(linestyles):
    axes[1].plot(x, y + n, color="blue", lw=2, linestyle=linestyle)
    
line, = axes[1].plot(x, y + 3, color="blue", lw=2)  # returns a list of line2D objects.
length1, gap1, length2, gap2 = 10, 7, 20, 7
line.set_dashes([length1, gap1, length2, gap2])   # 10 points + 7 points blank + 20 points + 7 points blank
axes_settings(fig, axes[1], "linetypes", len(linestyles) + 1)   # len(linestyles) + 1 is the ymax

markers = ['+', 'o', '*', 's', '.', '1', '2', '3', '4']
for n, marker in enumerate(markers):
    # lw = shorthand for linewidth, ls = shorthand for linestyle
    axes[2].plot(x, y + n, color="blue", lw=2, ls=':', marker=marker)
axes_settings(fig, axes[2], "markers", len(markers))

markersizecolors = [(4, "white"), (8, "red"), (12, "yellow"), (16, "lightgreen")]
for n, (markersize, markerfacecolor) in enumerate (markersizecolors):
    axes[3].plot(x, y + n, color="blue", lw=1, ls='-',
                marker='o', markersize=markersize,
                markerfacecolor=markerfacecolor,
                markeredgewidth=2)
axes_settings(fig, axes[3], "marker size/color", len(markersizecolors))
```


```python
sym_x = sympy.Symbol("x")
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)

def sin_expansion(x, n):
    '''Evaluate the nth order Taylor series expansion of sin(x) for the numerical values in the arry x'''
    return sympy.lambdify(sym_x, sympy.sin(sym_x).series(n=n+1).removeO(), 'numpy')(x)

fig, ax = plt.subplots()
ax.plot(x, np.sin(x), linewidth=4, color="red", label='exact')

colors = ["blue", "black"]
linestyles = [':', '-.', '--']
for idx, n in enumerate(range(1, 12, 2)):
    ax.plot(x, sin_expansion(x, n), color=colors[idx // 3],
            linestyle=linestyles[idx % 3], linewidth=3,
            label="order %d approx." % (n+1))
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(-1.5*np.pi, 1.5*np.pi)
# place a legend outsize of the Axes
ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
# make room for the legend to the right of the Axes
# fig.subplots_adjust(right=.75)
```

 Only lines with assigned labels are included in the legend (to assign a label to a line, use the label argument of Axes.plot
 see help(plt.legend) for details.

 bbox_to_anchor's argument is a tuple (x, y), where x and y are the canvas coordinates within the Axes object.
 (0, 0) correspondsto the lower-left corner, and (1, 1) corresponds to the upper-right corner.


 The default values can be set in the Matplotlib resource file, and session-wide configuration can be set in the mpl.rcParams
 dictionary. This dictionary is a cache of the Matplotlib resource file, and changes to parameters are valid until 
 the Python interpreter is restarted and Matplotlib is imported again. Try print(mpl.rcParams)


```python
x = np.linspace(0, 50, 500)
y = np.sin(x) * np.exp(-x/10)
fig, ax = plt.subplots(figsize=(8, 2), subplot_kw={'facecolor':
"#ebf5ff"})
ax.plot(x, y, lw=2)
ax.set_xlabel ("x", labelpad=5, fontsize=18, fontname='serif', color="blue")
ax.set_ylabel ("f(x)", labelpad=15, fontsize=18, fontname='serif', color="blue")
ax.set_title("axis labels and title example", fontsize=16, fontname='serif', color="blue")
```


```python
# set_xlim and set_ylim methods of the Axes object
# An alternative to set_xlim and set_ylim is the axis method
x = np.linspace(0, 30, 500)
y = np.sin(x) * np.exp(-x/10)
fig, axes = plt.subplots(1, 3, figsize=(9, 3), subplot_kw={'facecolor': "#ebf5ff"})

axes[0].plot(x, y, lw=2)
axes[0].set_xlim(-5, 35)
axes[0].set_ylim(-1, 1)
axes[0].set_title("set_xlim / set_y_lim")

axes[1].plot(x, y, lw=2)
axes[1].axis('tight')
axes[1].set_title("axis('tight')")

axes[2].plot(x, y, lw=2)
axes[2].axis('equal')
axes[2].set_title("axis('equal')")

# plt.Axes.set_yscale("log")  # this can change the scale 
```


```python
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
y = np.sin(x) * np.exp(-x**2/20)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

axes[0].plot(x, y, lw=2)
axes[0].set_title("default ticks")

axes[1].plot(x, y, lw=2)
axes[1].set_title("set_xticks")
axes[1].set_yticks([-1, 0, 1])
axes[1].set_xticks([-5, 0, 5])

axes[2].plot(x, y, lw=2)
axes[2].set_title("set_major_locator")
axes[2].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))   # the major tickers split the xaxis into 4 parts
axes[2].yaxis.set_major_locator(mpl.ticker.FixedLocator([-1, 0, 1]))
axes[2].xaxis.set_minor_locator(mpl.ticker.MaxNLocator(8))   # the minor tickers split the xaxis into 8 parts
axes[2].yaxis.set_minor_locator(mpl.ticker.MaxNLocator(8))

axes[3].plot(x, y, lw=2)
axes[3].set_title("set_xticklabels")
axes[3].set_yticks([-1, 0, 1])
axes[3].set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
axes[3].set_xticklabels([r'$-2\pi$', r'$-\pi$', 0, r'$\pi$',r'$2\pi$'])
x_minor_ticker = mpl.ticker.FixedLocator([-3 * np.pi / 2, -np.pi / 2, 0, np.pi / 2, 3 * np.pi / 2])
axes[3].xaxis.set_minor_locator(x_minor_ticker)
axes[3].yaxis.set_minor_locator(mpl.ticker.MaxNLocator(4))

# mpl.ticker module provides classes for different tick placement strategies
# mpl.ticker.MaxNLocator can be used to set the maximum number ticks
# mpl.ticker.MultipleLocator can be used for setting ticks at multiples of a given base 
# mpl.ticker.FixedLocator can be used to place ticks at explicitly specified coordinates
# use the set_major_locator and the set_minor_locator methods in Axes.xaxis and Axes.yaxis to change ticker strategy.
```


```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
x_major_ticker = mpl.ticker.MultipleLocator(4)
x_minor_ticker = mpl.ticker.MultipleLocator(1)
y_major_ticker = mpl.ticker.MultipleLocator(0.5)
y_minor_ticker = mpl.ticker.MultipleLocator(0.25)

for ax in axes:
    ax.plot(x, y, lw=2)
    ax.xaxis.set_major_locator(x_major_ticker)
    ax.yaxis.set_major_locator(y_major_ticker)
    ax.xaxis.set_minor_locator(x_minor_ticker)
    ax.yaxis.set_minor_locator(y_minor_ticker)

axes[0].set_title("default grid")
axes[0].grid()

axes[1].set_title("major/minor grid")
axes[1].grid(color="blue", which="both", linestyle=':', linewidth=0.5)

axes[2].set_title("individual x/y major/minor grid")
axes[2].grid(color="grey", which="major", axis='x', linestyle='-', linewidth=0.5)
axes[2].grid(color="grey", which="minor", axis='x', linestyle=':', linewidth=0.25)
axes[2].grid(color="grey", which="major", axis='y', linestyle='-', linewidth=0.5)



```


```python
# mpl.ticker module also provides classes for customizing the tick labels.
# ScalarFormatter in mpl.ticker module can set several useful properties related to displaying tick labels.
# If scientific notation is activated using the set_scientific method, use the set_ powerlimits method 
# (by default, tick labels for small numbers are not displayed using the scientific notation) to control the threshhold.
# use the useMathText=True argument to have the exponents shown in math style rather than code style eg: 1e10.

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
x = np.linspace(0, 1e5, 100)
y = x ** 2

axes[0].plot(x, y, 'b.')
axes[0].set_title("default labels", loc='right')

axes[1].plot(x, y, 'b')
axes[1].set_title("scientific notation labels", loc='right')

formatter = mpl.ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
axes[1].xaxis.set_major_formatter(formatter)
axes[1].yaxis.set_major_formatter(formatter)
```


```python
# log plot
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
x = np.linspace(0, 1e3, 100)
y1, y2 = x**3, x**4
axes[0].set_title('loglog')
axes[0].loglog(x, y1, 'b', x, y2, 'r')

axes[1].set_title('semilogy')
axes[1].semilogy(x, y1, 'b', x, y2, 'r')

axes[2].set_title('plot / set_xscale / set_yscale')
axes[2].plot(x, y1, 'b', x, y2, 'r')
axes[2].set_xscale('log')
axes[2].set_yscale('log')
```


```python
# The lines that make up the surrounding box are called axis spines in Matplotlib,
# we can use the Axes.spines attribute to change their properties.

x = np.linspace(-10, 10, 500)
y = np.sin(x) / x

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, linewidth=2)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# move bottom and left spine to x = 0 and y = 0
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))


ax.set_xticks([-10, -5, 5, 10])
ax.set_yticks([0.5, 1])
# give each label a solid background of white, to not overlap with the plot line
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_bbox({'facecolor': 'white', 'edgecolor': 'white'})
```


```python
# plt.figure, Figure.make_axes, and plt.subplots can create new Figure and Axes instances.
# The Figure.add_axes method is well suited for creating inset

fig = plt.figure(figsize=(8, 4))
def f(x):
    return 1/(1 + x**2) + 0.1/(1 + ((3 - x)/0.1)**2)

def plot_and_format_axes(ax, x, f, fontsize):
    ax.plot(x, f(x), linewidth=2)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.set_xlabel(r"$x$", fontsize=fontsize)
    ax.set_ylabel(r"$f(x)$", fontsize=fontsize)
# main graph    
ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], facecolor="#f5f5f5")
x = np.linspace(-4, 14, 1000)
plot_and_format_axes(ax, x, f, 18)

x0, x1 = 2.5, 3.5
ax.axvline(x0, ymax=0.3, color="grey", linestyle=":")
ax.axvline(x1, ymax=0.3, color="grey", linestyle=":")

ax_insert = fig.add_axes([0.5, 0.5, 0.38, 0.42], facecolor='none')
x = np.linspace(x0, x1, 1000)
plot_and_format_axes(ax_insert, x, f, 14)
```


```python
# plt.subplots is 鈥渟queezed鈥?by default: that is, the dimensions with length 1 are removed from the array. 
# If both the requested numbers of column and row are greater than one, then a two-dimensional array is returned.

# plt.subplots_adjust function allows to explicitly set the left, right, bottom, and top coordinates of the
# overall Axes grid, as well as the width (wspace) and height spacing (hspace) between Axes instances in the grid.
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True, squeeze=False)
x1 = np.random.randn(100)
x2 = np.random.randn(100)

axes[0, 0].set_title("Uncorrelated")
axes[0, 0].scatter(x1, x2)

axes[0, 1].set_title("Weakly positively correlated")
axes[0, 1].scatter(x1, x1 + x2)

axes[1, 0].set_title("Weakly negatively correlated")
axes[1, 0].scatter(x1, -x1 + x2)

axes[1, 1].set_title("Strongly correlated")
axes[1, 1].scatter(x1, x1 + 0.15 * x2)

axes[1, 1].set_xlabel("x")
axes[1, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[1, 0].set_ylabel("y")

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.1, hspace=0.2)
```


```python
# plt.subplot2grid function is an intermediary between plt.subplots and gridspec
# plt.subplot2grid takes two mandatory arguments: 
# the first argument is the shape of the Axes grid, in the form of a tuple (nrows, ncols),
# the second argument is a tuple (row, col) that specifies the starting position within the grid.
# two optional keyword arguments colspan and rowspan can be used to indicate 
# how many rows and columns the new Axes instance should span.
# plt.subplot2grid function results in one new Axes instance.

ax0 = plt.subplot2grid((3, 3), (0, 0))
ax1 = plt.subplot2grid((3, 3), (0, 1))
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
```


```python
# A GridSpec object is only used to specify the grid layout, and by itself it does not create any Axes objects. 
# When creating a new instance of the GridSpec class, we must specify the number of rows and columns in the grid.
fig = plt.figure(figsize=(6, 4))
gs = mpl.gridspec.GridSpec(4, 4)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[2, 2])
ax3 = fig.add_subplot(gs[3, 3])
ax4 = fig.add_subplot(gs[0, 1:])
ax5 = fig.add_subplot(gs[1:, 0])
ax6 = fig.add_subplot(gs[1, 2:])
ax7 = fig.add_subplot(gs[2:, 1])
ax8 = fig.add_subplot(gs[2, 3])
ax9 = fig.add_subplot(gs[3, 2])
```


```python
# use pcolor, imshow, contour and contourf functions to graph data.
# predefined color maps in Matplotlib are available in mpl.cm. Try help(mpl.cm)
# the vmin and vmax can be used to set the range of values that are mapped to the color axis. 
# This can equivalently be achieved by setting norm=mpl.colors.Normalize(vmin, vmax).




x = y = np.linspace(-10, 10, 150)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) * np.cos(Y) * np.exp(-(X/5)**2-(Y/5)**2)

fig, ax = plt.subplots(figsize=(6, 5))
norm = mpl.colors.Normalize(-abs(Z).max(), abs(Z).max())
p = ax.pcolor(X, Y, Z, norm=norm, cmap=mpl.cm.bwr)

ax.axis('tight')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

cb = fig.colorbar(p, ax=ax)
cb.set_label(r"$z$", fontsize=18)
cb.set_ticks([-1, -.5, 0, .5, 1])
```


```python
# drawing 3D graphs requires using a different axes object, 
# the Axes3D object that is available from the mpl_toolkits.mplot3d module
# ax = Axes3D(fig) or use the add_subplot function with the projection='3d' argument:
# ax = ax = fig.add_subplot(1, 1, 1, projection='3d') or use plt.subplots with the subplot_kw={'projection': '3d'} argument:
# fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': '3d'}).

# the Axes3D class methods 鈥搒uch as plot_surface, plot_wireframe, and contour 
# can be used to plot data as surfaces in a 3D perspective.

# plot_surface function takes rstride and cstride (row and column stride) for selecting data from the input arrays.
# contour and contourf functions take optional arguments zdir and offset, which is used to select a projection direction.

# In addition to the methods for 3D surface plotting, 
# plot, scatter, bar, and bar3d are available in the Axes3D class takes an additional argument for the z coordinates.
# Like their 2D relatives, these functions expect one-dimensional data arrays rather than the two-dimensional
# coordinate arrays that are used for surface plots.


fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={'projection': '3d'})

def title_and_labels(ax, title):
    ax.set_title(title)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    ax.set_zlabel("$z$", fontsize=16)

x = y = np.linspace(-3, 3, 74)
X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(4 * R) / R

norm = mpl.colors.Normalize(-abs(Z).max(), abs(Z).max())
p = axes[0].plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, norm=norm, cmap=mpl.cm.Blues)
cb = fig.colorbar(p, ax=axes[0], shrink=0.6)
title_and_labels(axes[0], "plot_surface")

p = axes[1].plot_wireframe(X, Y, Z, rstride=2, cstride=2, color="darkgrey")
title_and_labels(axes[1], "plot_wireframe")

cset = axes[2].contour(X, Y, Z, zdir='z', offset=0, norm=norm, cmap=mpl.cm.Blues)   # projection on z index
cset = axes[2].contour(X, Y, Z, zdir='y', offset=3, norm=norm, cmap=mpl.cm.Blues)   # projection on y index
title_and_labels(axes[2], "contour")
```


```python
# some recommendations:
# Devert, A. (2014). matplotlib Plotting Cookbook. Mumbai: Packt
# J. Steele, N. I. (2010). Beautiful Visualization. Sebastopol: O'Reilly
# Milovanovi, I. (2013). Python Data Visualization Cookbook. Mumbai: Packt
```
