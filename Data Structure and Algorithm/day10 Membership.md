
## 1. Bloom Filters
In some cases the word processor may suggest an alternative, correctly spelled word. In some cases, the word processor may simply correct the misspelling. How is this funciton implemented?  

__Bloom filter__ is an array of bits along with a set of hashing functions. The number of bits in the filter and the number of hashing functions influences the accuracy of the bloom filter. This data structure uses statistical probability to determine if an item is a member of a set of valuse. It's not 100% accurate, it will __never report a false negative__, meaning they will never report an item doesn't belong to a set when it actually does. It will sometimes report a __false positive__. It may report an item is in a set when it is actually not.  

Consider a bloom filter with 20 bits and 3 independent hash functions. Initially all the bits in the filter are set to 0. If we add the word _cow_ to our filter, use the three independent hash funciton to hash _cow_, then modulo 20(20bits), assume we got 18, 9, 3. Then we set the indices 18, 9, 3 to 1, so _cow_ is in the fiter now.  
Looking up an item in a bloom filter requires hashing the value again with the same hash functions generating the indices into the bit array. If the value at all indices in the bit array are one, then the lookup function reports success and otherwise failure.  

If a bloom filter is to be useful, it must never report a false negative and false positives must be kept to a minimum. It is possible to determine on average how often a bloom filter will report a false positive. This probability depends on:
1. the hashing functions
2. the number of items added to the bloom filter 
3. the number of bits used in the bloom filter

### The hashing funcitons
Two requirements must be complied: 
1. completely independent of each other.
2. Each hashing function must also be evenly distributed over the range of bit indices in the bit array.

Uniform distribution is guaranteed by the built-in hash functions and some hashing funcitons allow a seed value for creating different hashing functions. Another way of generating independent hashing functions is to append some known value to the end of each item before it is hashed, like _rabbit0, rabbit1_. 
### The Bloom fiter size
We can find the BF size with a number of items to insert and a desired false positive probability. Assume the BF consists of __m bits__锛?
1. the probaility(p) of any one location within a BF not being set by a hash function while inserting an item is ${1}-\frac{1}{m}$.
2. if the BF uses k hash funcitons, the p that a bit in the BF is not set when inserting an item is $\left ( {1}-\frac{1}{m} \right )^{k}$.
3. if n items are inserted, the p that a bit in the BF is still zero after inserting all __n items__ can be: $\left ( {1}-\frac{1}{m} \right )^{nk}$.
4. So, the p that a bit in the BF is a 1 after inserting n items is $ {1}-\left ( {1}-\frac{1}{m} \right )^{nk}$.
5. If we look up an item that was not added to the BF, each time we hash the item we get 1 for k hashing functions means a bit in the BF is one for k times, this p is the p that the BF will report a false positive: $ \left ({1}-\left ( {1}-\frac{1}{m} \right )^{nk}  \right )^{k}$.
6. this formula can be approximated as $\left ( 1 -e^{\frac{kn}{m}}\right )^{k}$  

Given _n_ and _p_, we can got $m=-\frac{n\ln p}{\left (ln 2  \right )^{2}}$ and $k=\frac{m}{n}\ln 2$  

## 2. Trie Datatype
A trie is not meant to be used when deleting values from a data structure is required. It's only for retrieval of items based on a key value.  
Tries are appropriate when key values are made up of more than one unit and when the individual units of a key may overlap with other item keys. Words are made up of characters, these characters are the individual units of the keys. Many words overlap in a dictionary like a, an, and ant. The linked trie which is a series of link lists making up a matrix.  

The trie data structure begins with an empty linked list. Each node in the linked trie list contains three values: 
1. a unit of the key (in the spellchecker instance this is a character of the word).
2. a next pointer that points to the next node in the list which would contain some other unit appearing at the same position within
a key(like in _ant, and_ node(__t__)'s next pointer point to __d__)
3. a follows pointer which points at a node that contains the next unit within the same key.

When items are inserted into the trie a __sentinel unit__ is added. In the case of the spell checker, a 鈥?鈥?character is appended to the end of every word. The sentinel is needed because words like rat are prefixes to words like ratchet.
