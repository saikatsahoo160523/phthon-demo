import numpy as np

# conver 1-d array to numpy array
arry1 = [12,67,34]
np_arry1 = np.array(arry1)
print(np_arry1)

# conver 2-d array to numpy array
arry2 = [ [4,5,5],[7,2,2],[67,45,7] ]
np_arry2 = np.array(arry2)
print(np_arry2)

# to access data elements - can use matrix notation
print(np_arry2[2,1])

# np_arry3 = np.matrix(arry2)
# print(np_arry3)

# to check the number of rows and colloms
print(np_arry2.shape())
# to check the total number of element
print(np_arry2.size)
# can perform matrix operation
print(np_arry2.T)
print(np_arry2.min())
print(np_arry2.max())
print(np_arry2.sum())
print(np_arry2.sum(axis=0))
print(np_arry2.sum(axis=1))
# to check dimension detx
print(np_arry2.ndim)
# to check data-typ
print(np_arry2.dtype)
print(np_arry1.dtype)


#can initialize 0  with row X col
np_arry4=np.zeros((2,3))
print(np_arry4)

#can initialize 1  with row X col = 2 x 3
np_arry5=np.ones((2,3))
print(np_arry5)
# or
np_arry6=np.empty((2,3))
print(np_arry6)

# reshape and return as new variable
# it is not possible to change the original
# use -1 for auto adjust
print(np_arry2.reshape(3,-1))

#Spliting - horizontal format & vertical format
print(np.hsplit(np_arry2,3))

##############################################


# Python program to demonstrate
# basic array characteristics
import numpy as np

# Creating array object
arr = np.array( [[ 1, 2, 3],
				[ 4, 2, 5]] )

# Printing type of arr object
print("Array is of type: ", type(arr))

# Printing array dimensions (axes)
print("No. of dimensions: ", arr.ndim)

# Printing shape of array
print("Shape of array: ", arr.shape)

# Printing size (total number of elements) of array
print("Size of array: ", arr.size)

# Printing type of elements in array
print("Array stores elements of type: ", arr.dtype)


##############################################

# Python program to demonstrate
# array creation techniques
import numpy as np

# Creating array from list with type float
a = np.array([[1, 2, 4], [5, 8, 7]], dtype='float')
print("Array created using passed list:\n", a)

# Creating array from tuple
b = np.array((1, 3, 2))
print("\nArray created using passed tuple:\n", b)

# Creating a 3X4 array with all zeros
c = np.zeros((3, 4))
print("\nAn array initialized with all zeros:\n", c)

# Create a constant value array of complex type
d = np.full((3, 3), 6, dtype='complex')
print("\nAn array initialized with all 6s."
      "Array type is complex:\n", d)

# Create an array with random values
e = np.random.random((2, 2))
print("\nA random array:\n", e)

# Create a sequence of integers
# from 0 to 30 with steps of 5
f = np.arange(0, 30, 5)
print("\nA sequential array with steps of 5:\n", f)

# Create a sequence of 10 values in range 0 to 5
g = np.linspace(0, 5, 10)
print("\nA sequential array with 10 values between"
      "0 and 5:\n", g)

# Reshaping 3X4 array to 2X2X3 array
arr = np.array([[1, 2, 3, 4],
                [5, 2, 4, 2],
                [1, 2, 0, 1]])

newarr = arr.reshape(2, 2, 3)

print("\nOriginal array:\n", arr)
print("Reshaped array:\n", newarr)

# Flatten array
arr = np.array([[1, 2, 3], [4, 5, 6]])
flarr = arr.flatten()

print("\nOriginal array:\n", arr)
print("Fattened array:\n", flarr)



##############################################

# Python program to demonstrate
# indexing in numpy
import numpy as np

# An exemplar array
arr = np.array([[-1, 2, 0, 4],
                [4, -0.5, 6, 0],
                [2.6, 0, 7, 8],
                [3, -7, 4, 2.0]])

# Slicing array
temp = arr[:2, ::2]
print("Array with first 2 rows and alternate"
      "columns(0 and 2):\n", temp)

# Integer array indexing example
temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]]
print("\nElements at indices (0, 3), (1, 2), (2, 1),"
      "(3, 0):\n", temp)

# boolean array indexing example
cond = arr > 0  # cond is a boolean array
temp = arr[cond]
print("\nElements greater than 0:\n", temp)


##############################################


# Python program to demonstrate
# basic operations on single array
import numpy as np

a = np.array([1, 2, 5, 3])

# add 1 to every element
print("Adding 1 to every element:", a + 1)

# subtract 3 from each element
print("Subtracting 3 from each element:", a - 3)

# multiply each element by 10
print("Multiplying each element by 10:", a * 10)

# square each element
print("Squaring each element:", a ** 2)

# modify existing array
a *= 2
print("Doubled each element of original array:", a)

# transpose of array
a = np.array([[1, 2, 3], [3, 4, 5], [9, 6, 0]])

print("\nOriginal array:\n", a)
print("Transpose of array:\n", a.T)


##############################################

# Python program to demonstrate
# unary operators in numpy
import numpy as np

arr = np.array([[1, 5, 6],
                [4, 7, 2],
                [3, 1, 9]])

# maximum element of array
print("Largest element is:", arr.max())
print("Row-wise maximum elements:",
      arr.max(axis=1))

# minimum element of array
print("Column-wise minimum elements:",
      arr.min(axis=0))

# sum of array elements
print("Sum of all array elements:",
      arr.sum())

# cumulative sum along each row
print("Cumulative sum along each row:\n",
      arr.cumsum(axis=1))

##############################################

# Python program to demonstrate
# binary operators in Numpy
import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[4, 3],
              [2, 1]])

# add arrays
print("Array sum:\n", a + b)

# multiply arrays (elementwise multiplication)
print("Array multiplication:\n", a * b)

# matrix multiplication
print("Matrix multiplication:\n", a.dot(b))

##############################################

# Python program to demonstrate
# universal functions in numpy
import numpy as np

# create an array of sine values
a = np.array([0, np.pi / 2, np.pi])
print("Sine values of array elements:", np.sin(a))

# exponential values
a = np.array([0, 1, 2, 3])
print("Exponent of array elements:", np.exp(a))

# square root of array values
print("Square root of array elements:", np.sqrt(a))

##############################################

# Python program to demonstrate sorting in numpy
import numpy as np

a = np.array([[1, 4, 2],
              [3, 4, 6],
              [0, -1, 5]])

# sorted array
print("Array elements in sorted order:\n",
      np.sort(a, axis=None))

# sort array row-wise
print("Row-wise sorted array:\n",
      np.sort(a, axis=1))

# specify sort algorithm
print("Column wise sort by applying merge-sort:\n",
      np.sort(a, axis=0, kind='mergesort'))

# Example to show sorting of structured array
# set alias names for dtypes
dtypes = [('name', 'S10'), ('grad_year', int), ('cgpa', float)]

# Values to be put in array
values = [('Hrithik', 2009, 8.5), ('Ajay', 2008, 8.7),
          ('Pankaj', 2008, 7.9), ('Aakash', 2009, 9.0)]

# Creating array
arr = np.array(values, dtype=dtypes)
print("\nArray sorted by names:\n",
      np.sort(arr, order='name'))

print("Array sorted by grauation year and then cgpa:\n",
      np.sort(arr, order=['grad_year', 'cgpa']))

##############################################

import numpy as np

a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

# vertical stacking
print("Vertical stacking:\n", np.vstack((a, b)))

# horizontal stacking
print("\nHorizontal stacking:\n", np.hstack((a, b)))

c = [5, 6]

# stacking columns
print("\nColumn stacking:\n", np.column_stack((a, c)))

# concatenation method
print("\nConcatenating to 2nd axis:\n", np.concatenate((a, b), 1))

##############################################


import numpy as np

a = np.array([[1, 3, 5, 7, 9, 11],
              [2, 4, 6, 8, 10, 12]])

# horizontal splitting
print("Splitting along horizontal axis into 2 parts:\n", np.hsplit(a, 2))

# vertical splitting
print("\nSplitting along vertical axis into 2 parts:\n", np.vsplit(a, 2))

##############################################


import numpy as np

a = np.array([1.0, 2.0, 3.0])

# Example 1
b = 2.0
print(a * b)

# Example 2
c = [2.0, 2.0, 2.0]
print(a * c)


##############################################

import numpy as np

a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([0.0, 1.0, 2.0])

print(a[:, np.newaxis] + b)


##############################################

import numpy as np

# creating a date
today = np.datetime64('2017-02-12')
print("Date is:", today)
print("Year is:", np.datetime64(today, 'Y'))

# creating array of dates in a month
dates = np.arange('2017-02', '2017-03', dtype='datetime64[D]')
print("\nDates of February, 2017:\n", dates)
print("Today is February:", today in dates)

# arithmetic operation on dates
dur = np.datetime64('2017-05-22') - np.datetime64('2016-05-22')
print("\nNo. of days:", dur)
print("No. of weeks:", np.timedelta64(dur, 'W'))

# sorting dates
a = np.array(['2017-02-12', '2016-10-13', '2019-05-22'], dtype='datetime64')
print("\nDates in sorted order:", np.sort(a))


##############################################

import numpy as np

A = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])

print("Rank of A:", np.linalg.matrix_rank(A))

print("\nTrace of A:", np.trace(A))

print("\nDeterminant of A:", np.linalg.det(A))

print("\nInverse of A:\n", np.linalg.inv(A))

print("\nMatrix A raised to power 3:\n", np.linalg.matrix_power(A, 3))


##############################################

import numpy as np

# coefficients
a = np.array([[1, 2], [3, 4]])
# constants
b = np.array([8, 18])

print("Solution of linear equations:", np.linalg.solve(a, b))

##############################################

import numpy as np
import matplotlib.pyplot as plt

# x co-ordinates
x = np.arange(0, 9)
A = np.array([x, np.ones(9)])

# linearly generated sequence
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
# obtaining the parameters of regression line
w = np.linalg.lstsq(A.T, y)[0]

# plotting the line
line = w[0] * x + w[1]  # regression line
plt.plot(x, line, 'r-')
plt.plot(x, y, 'o')
plt.show()

##############################################




