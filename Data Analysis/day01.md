

```python
# some books need to follow up in the future
Python Cookbook, Third Edition, by David Beazley and Brian K. Jones 
Fluent Python by Luciano Ramalho 
Effective Python by Brett Slatkin (Pearson)

# ploty, seaborn, Bokeh
```


```python
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

%matplotlib notebook  # makes the figures interactive
```

## 1.Numpy Basics


```python
data = {i : np.random.randn() for i in range(7)} # remember generator
data = np.random.randn(2, 3) # remember those basic functions in np.random packages

# 1.creating ndarrys
arr1 = np.array([0,1,2,3])    # this is one dimentional
print(arr1.ndim)
arr2 = np.array([[0,1,2,3]])  # this is two dimentional
print(arr2.ndim)

# 2.change the data types
arr2.astype(np.float64)
arr2.dtype
# type(arr2) will not show the data type but only show ndarry
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_) # we can specify the dtype when we creat an array.

# 3.An important first distinction from Python鈥檚 built-in lists is that array slices are views on the original array.
# This means that the data is not copied, and any modifications to the view will be reflected in the source array.

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
# see the out puts to find out which is index-0 and which is index-1
arr2d[:2] # this is slicing the index-0
arr2d[:][1:] # this is slicing the index-1

names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
data[names == 'Bob', 2:] # this will use boolean index select from rows and use slices index the columns
# 4.boolean index can be combined with &(and), |(or)


# 5.fancy index
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr[[4,3,0,6]]
arr[[-3,-5,-7]]
arr[[1, 5, 7, 2],[0, 3, 1, 2]]
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

# 6.Element-Wise fun
# np.sqrt()
# np.exp()
# np.maximum()
# np.modf()
# for more fun, need to check documentation

# 7. array-oriented programming
# The np.meshgrid function takes two 1D arrays and produces two 2D matrices corresponding to all pairs of (x, y) in the two arrays.
points = np.arange(-5,5,0.01)
xs, ys = np.meshgrid(points, points)
# Expressing conditional logics as array operations
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
# traditional method is not very fast
result = [(x if c else y)
         for x, y, c in zip(xarr, yarr, cond)]
# array-oriented method is fast
result = np.where(cond, xarr, yarr)
np.where(arr > 0, 2, arr)

# math ans stat fun 
arr = np.random.randn(5,4)
arr.mean(axis=1)
arr.sum(axis=0)
arr.cumsum(axis=0)
arr.cumprod(axis=1)
# argmin, argmax, std, var, these funs you can explore with some matrixs

large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]    # 5% quantile

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])
# unique(x), intersect1d(x, y), union1d(x, y), in1d(x, y), setdiff1d(x, y), setxor1d(x, y)


# Random Walks V1.0
import random
import matplotlib.pyplot as plt
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
    
plt.plot(walk[:100])


# V2.0
nsteps = 1000
draws = np.random.randint(0,2,size=nsteps)
steps = np.where(draws>0, 1, -1)
walk = steps.cumsum()
walk.min()
walk.max()
(np.abs(walk) >= 10).argmax()

# Simulating Many Random Walks at Once
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
```

## 2. Start with Pandas


```python
# 1.The column returned from indexing a DataFrame is a view on the
# underlying data, not a copy. Thus, any in-place modifications to the
# Series will be reflected in the DataFrame.

pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
DataFrame(pop, index=[2001,2002,2003])

# 2.index objects is immutable. some index methods: append, difference, intersection, union, isin, delete, drop, insert, is_monotonic
# is_unique, unique

# 3.Reindexing, reindex is an important method, use TAB to explore it.
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b':'c']  # this is different, the 'b' and 'c' will remain, have a try

# 4.df[val]  special case: boolean array(filter rows), slice(slice rows)

# broadcasting 
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame - series
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
frame + series2

# 5.Function Application and Mapping 
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
f = lambda x: x.max() - x.min()
frame.apply(f)
frame.apply(f, axis='columns')

def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)

# Element-wise fun use applymap
formater = lambda x: '%.2f'% x
frame.applymap(formater)


# 6.sorting and ranking
frame = pd.DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame.sort_index(axis=1, ascending=False)

frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by='b')
frame.sort_values(by=['a', 'b'])

obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()  # there are some params in this method rank, explore when needed

df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
df.sum(axis='columns')
df.mean(axis='columns', skipna=False)

# idxmin and idxmax, return indirect statistics like the index value where the minimum or maximum values are attained
df.idxmax()
# DataFrame.idxmax(), use TAB explore the params

# On non-numeric data, describe produces alternative summary statistics
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
```

## 3.Data Cleaning and Preparation


```python
# 1. hanle missing data, methods like: isnull, notnull, dropna, fillna
# Calling fillna with a dict, you can use a different fill value for each column

# 2. remove duplicates: drop_duplicates(), duplicated()
# 3. use a fun or mapping to transform data
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef', 'Bacon', 
                    'pastrami', 'honey ham', 'nova lox'], 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {'bacon': 'pig', 'pulled pork': 'pig', 
                  'pastrami': 'cow', 'corned beef': 'cow',
                  'honey ham': 'pig', 'nova lox': 'salmon'}
lowercased = data['food'].str.lower()
data['animal'] = lowercased.map(meat_to_animal)
data['food'].map(lambda x: meat_to_animal[x.lower()])

# 4. replacing values
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace(-999, np.nan)
data.replace([-999, -1000], np.nan)
data.replace([-999, -1000], [np.nan, 0])
data.replace({-999: np.nan, -1000: 0})

# 5. renaming axis indexes
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
transform = lambda x: x[:4].upper()
data.index = data.index.map(transform)
data
data.rename(index=str.title, columns=str.upper)
data.rename(index={'OHIO':'INDIANA'}, columns={'three':'peekaboo'}, inplace=True)

# 6. Discretization and Binning
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins, right=False)
cats.codes
cats.categories
pd.value_counts(cats)
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)
data = np.random.rand(20)
pd.cut(data,4,precision=2)   # use the equal-length bins computed from the max and min of the data to cut the data into 4 parts

data = np.random.randn(1000)
cats = pd.qcut(data, 4)
pd.value_counts(cats)
pd.qcut(data, [0,0.1,0.5,0.9,1.]).categories

# 7. Permutation and random sampling
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
df.take(sampler)

df.sample(n=3)
choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)

df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
pd.get_dummies(df['key'])

# Vectorized string fun
import re

data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = Series(data)
data.str.contains('gmail')
pattern = '([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\\.([A-Z]{2,4})'
data.str.findall(pattern, flags=re.IGNORECASE)
```

## 4. Data Wrangling


```python
# 1. Hierachical indexing
data = pd.Series(np.random.randn(9),
                index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data.unstack()
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                    columns=[['Ohio', 'Ohio', 'Colorado'],
                            ['Green', 'Red', 'Green']])
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame.swaplevel('key1','key2')
frame.sort_index(level=1)
frame.swaplevel(0, 1).sort_index(level=0)

frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                    'c': ['one', 'one', 'one', 'two', 'two',
                        'two', 'two'],
                    'd': [0, 1, 2, 0, 1, 2, 3]})
frame.set_index(['c', 'd'], drop=False)    # By default the columns are removed from the DataFrame, though you can leave them in.
# reset_index(),does the opposite of set_index()

# 2. combine and merge datasets
# pandas.merge connects rows in df based on one or more keys.
# pandas.concat stacks together objects along an axis.

# combine_first is an instance methond, 
# enables splicing together overlapping data to fill in missing values in an obj with values from another.
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, how='outer')

df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')

left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
                    'key2': ['one', 'two', 'one'],
                    'lval': [1, 2, 3]})
right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                    'key2': ['one', 'one', 'one', 'two'],
                    'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))

# Merging on index:
# when the merge keys in a DF will be found in its index, 
# you can pass left_index=True or right_index=True to indicate that the index
# should be used as the merge keys
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
pd.merge(left1, right1, left_on='key', right_index=True)

lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio',
                                'Nevada', 'Nevada'],
                     'key2': [2000, 2001, 2002, 2001, 2002],
                     'data': np.arange(5.)})
righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
                     index=[['Nevada', 'Nevada', 'Ohio', 'Ohio',
                             'Ohio', 'Ohio'],
                             [2001, 2000, 2000, 2000, 2001, 2002]],
                     columns=['event1', 'event2'])

pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')

# concatenate along an axis
arr = np.arange(12).reshape((3, 4))
np.concatenate([arr, arr], axis=1)

s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
pd.concat([s1, s2, s3], axis=1)
s4 = pd.concat([s1, s3])
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                    columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                    columns=['three', 'four'])
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
pd.concat({'level1': df1, 'level2': df2}, axis=1)
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower'])

# combining data with overlap
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
np.where(pd.isnull(a), b, a)

# combine_first patching missing data in the calling obj with data from the obj passed to it, column by column.
df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
                    'b': [np.nan, 2., np.nan, 6.],
                    'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],
                    'b': [np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2)

# 3. reshaping and pivoting
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],
                    name='number'))
result = data.stack()
result.unstack() # By default the innermost level is unstacked (same with stack).
# You can unstack a different level by passing a level number or name.
result.unstack('state')

s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])  
# remember this keys params, its useful, we have discussed this is the combine parts
data2.unstack()
data2.unstack().stack()  # Stacking filters out missing data by default.
data2.unstack().stack(dropna=False)

# pivot is an useful fun, use TAB to explore more.
df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                    'A': [1, 2, 3],
                    'B': [4, 5, 6],
                    'C': [7, 8, 9]})
melted = pd.melt(df, ['key'])  # use TAB explore this fun, it is used to pivot wide to long format.
# key column is a group indicator
reshaped = melted.pivot('key', 'variable', 'value')
reshaped.reset_index()    # this will make the 'key' out of index and use it as a column.

# You can also specify a subset of columns to use as value columns.
pd.melt(df, id_vars=['key'], value_vars=['A', 'B'])
pd.melt(df, value_vars=['A', 'B', 'C'])
pd.melt(df, value_vars=['key', 'A', 'B'])
```

## 5. Plotting and Visualization


```python
data = np.arange(10)
# plt.plot(data)


# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# plt.plot(np.random.randn(50).cumsum(), 'k--')
# _ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
# ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))

# fig, axes = plt.subplots(2, 3), there are many params in this fun, use TAB to explore.
# axes

# Adjusting the spacing around subplots:          like sharex, sharey, subplot_kw
# use the subplots_adjust method on Figure obj. this method is also available as top-level fun.
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# for i in range(2):
#     for j in range(2):
#         axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
# plt.subplots_adjust(wspace=0, hspace=0)

# Ticks, Labels, and Legends:
# there are two ways to do plot decorations: 1.use the pyplot interface, like matplotlib.pyplot. 2. use obj-oriented API.
# xlim, xticks, xticklabels, all these methods act on the most recently created AxesSubplot. plt.xlim([0, 10])
# you can also use the methods on the subplot obj itself, like ax.get_xlim, ax.set_xlim.

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(np.random.randn(1000).cumsum())
# ticks = ax.set_xticks([0,250,500,750,1000])
# labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
# ax.set_title('My first matplotlib plot')
# ax.set_xlabel('Stages')

# # The axes class has a set method that allows batch setting of plot properties.
# props = {
#     'title': 'My first matplotlib plot',
#     'xlabel': 'Stages'
# }
# ax.set(**props)   # use **props to pass in keywords arguments.
# ax.legend(loc='best')

# plt.savefig('figpath.png', dpi=400, bbox_inches='tight')   # use TAB explore this function.

# plt.rc('figure', figsize=(10, 10))
# font_options = {'family' : 'monospace',
#                 'weight' : 'bold',
#                 'size' : 'small'}
# plt.rc('font', **font_options)

# Plotting with pandas and seaborn
# pandas has builit-in methods to visualize DF and Series.
# s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
# s.plot()   # use TAB explore this fun, many params to adjust the figure, like use_index, xticks,xlim....

# DF plot method plots each of its columns as a different line on the same subplot
# df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
#                     columns=['A', 'B', 'C', 'D'],
#                     index=np.arange(0, 100, 10))
# df.plot()   
# The plot attribute contains a 鈥渇amily鈥?of methods for different plot types. 
# For example, df.plot() is equivalent to df.plot.line().

# fig, axes = plt.subplots(2, 1)
# data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
# data.plot.bar(ax=axes[0], color='k', alpha=0.7)
# data.plot.barh(ax=axes[1], color='k', alpha=0.7)

# With a DataFrame, bar plots group the values in each row together in a group in bars.
# df = pd.DataFrame(np.random.rand(6, 4),
#                     index=['one', 'two', 'three', 'four', 'five', 'six'],
#                     columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
# df.plot.bar()
# df.plot.barh(stacked=True, alpha=0.5)
```

## 6. Data Aggregation and Group Operations


```python
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                    'key2' : ['one', 'two', 'one', 'two', 'one'],
                    'data1' : np.random.randn(5),
                    'data2' : np.random.randn(5)})
grouped = df['data1'].groupby(df['key1'])
grouped.mean()
means = df['data1'].groupby([df['key1'], df['key2']]).mean()

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()
df.groupby(['key1', 'key2']).size()
# Take note that any missing values in a group key will be excluded from the result.
# for name, group in df.groupby('key1'):
#     print(name)
#     print(group)
    
# for (k1, k2), group in df.groupby(['key1', 'key2']):
#     print((k1, k2))
#     print(group)
    
pieces = dict(list(df.groupby('key1')))      # groupby will generate tuple, so we use list() to convert it first
pieces['b']

# By default groupby groups on axis=0, but you can group on any of the other axes.
df.dtypes
df
grouped = df.groupby(df.dtypes, axis=1)
# for dtype, group in grouped:
#     print(dtype)
#     print(group)

# Selecting a Column or Subset of Columns
df.groupby('key1')['data1']
df.groupby('key1')[['data2']]

df['data1'].groupby(df['key1'])
df[['data2']].groupby(df['key1'])
# these four lines are the same
df.groupby(['key1', 'key2'])[['data2']].mean()

# Grouping with Dicts and Series
people = pd.DataFrame(np.random.randn(5, 5),
                        columns=['a', 'b', 'c', 'd', 'e'],
                        index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.iloc[2:3, [1, 2]] = np.nan
mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f' : 'orange'}
by_column = people.groupby(mapping, axis=1)
by_column.sum()

map_series = pd.Series(mapping)
people.groupby(map_series, axis=1).count()

# Grouping with Functions
people.groupby(len).sum() # use the lengths of the strings in the index to group
key_list = ['one', 'one', 'one', 'two', 'two']
# Mixing functions with arrays, dicts, or Series is not a problem as everything gets converted to arrays internally.
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()


# Grouping by Index Levels
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'], [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
hier_df.groupby(level='cty', axis=1).count()

# Data Aggregation     count, sum, mean, median, std, var, min, max, prod, first, last
df
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)

def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)

grouped.describe()

# Column-Wise and Multiple Function Application
grouped_pct.agg(['mean', 'std', peak_to_peak])  # after you grouped sth, you can call agg() method on it, and TAB to explore it.

# pass a list of (name, function) tuples, the first element of each tuple will be used as the DataFrame column names
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

# a list of tuples with custom names can be passed
ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)  # this looks like the (name, function) tuples where the name is DF column names.
# but it is different, since we used the column names first, then we call agg(), this produces nicknames for the functions in df. 

# when we wanted to apply several different functions to one or more of the columns
grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'], 'size' : 'sum'})

# the aggregated data comes back with an index, potentially hierarchical, composed from the unique group key combinations.
# if we want disable this property. we passing in as_index=False
tips.groupby(['day', 'smoker'], as_index=False).mean()








```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key1</th>
      <th>key2</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>one</td>
      <td>-0.628617</td>
      <td>0.441954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>two</td>
      <td>-0.973279</td>
      <td>0.516811</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>one</td>
      <td>-1.045809</td>
      <td>-0.279760</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>two</td>
      <td>0.335970</td>
      <td>-0.123966</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>one</td>
      <td>0.661918</td>
      <td>0.648523</td>
    </tr>
  </tbody>
</table>
</div>




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


```python

```
