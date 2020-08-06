
## 1. The behavior of RAM
1. RAM of a computer does not behave like a post office.  
The computer does not need to find the right RAM location before the it can retrieve or store a value.  
2. RAM is like a group of people, each person is assigned an address.  
It does not take any time to find the right person because all the people are listening, just in case their name is called.  

All locations within the RAM of a computercan be accessed in the same amount of time.  
So this characteristic leads two results in python and we can test them:  

__a__. The size of a list does not affect the average access time in the list.  
__b__. The average access time at any location within a list is the same, regardless of its location within the list.


```python
import datetime
import random
import time

# the size of test lists range from 1000 to 200000
xmin = 1000  # sizemin
xmax = 200000  # sizemax

# record the list sizes in sizeList
# record average access time of a list for 1000 retrievals in timeList
xList = []  # sizeList
yList = []  # timeList

for x in range(xmin, xmax+1,1000):
    xList.append(x)
    prod = 0
    # creat a list of 0s, size is x (1000,2000,....)
    lst = [0]*x
    
    starttime = datetime.datetime.now()

    for v in range(1000):
        index = random.randint(0,x-1)
        val = lst[index]
        # dummy operation
        prod = prod*val
    endtime = datetime.datetime.now()

    delatT = endtime - starttime
    # Divide by 1000 for the average access time, but also multiply by 1000000 for microseconds
    accessTime = delatT.total_seconds()*1000
    yList.append(accessTime)

spec = max(yList) - min(yList)  # this is the variation in microseconds

```
