#https://www.w3resource.com/python-exercises/numpy
import numpy as np
import matplotlib.pyplot as plt

#Basic exercises

# x1 = np.arange(9.0).reshape((3, 3))
# print(x1)
# x2 = np.arange(3.0)
# print(x2)
# print(np.add(x1, x2))

# x3 = np.trunc(np.random.randn(10))
# print(~np.all(x3))
# print(np.any(x3))

# x4 = np.abs(np.trunc(np.random.randn(10)*10))
# x5 = np.trunc(np.random.randn(10))
# x6 = x5/x4
# print(x6)
# print(np.isinf(x6))

# x7 = np.array([np.zeros(10),np.full(10,10),np.full(10,5)])
# print(x7)

# x8 = np.array([i for i in range(30, 71)])
# x8 = np.arange(30,71)
# print(x8)
# x9 = x8[(x8%2)==0]
# x9 = np.arange(30,71,2)
# print(x9)

# x10 = np.identity(3)
# print(x10)

# x11 = np.arange(15,56)
# print(x11[1:-1])

# x12 = np.arange(0,12).reshape(3,4)
# print(x12)
# for i in np.nditer(x12):
#     print(i, end=' ')

# x13 = np.linspace(5,50,10)
# print(x13)

# x14 = np.arange(0,21)
# x14[(x14>=9) & (x14<=15)]*=-1
# print(x14)

# x15 = np.random.randint(0, 11, 5)
# print(x15)

# x16 = np.arange(10,22).reshape(3,4)
# print(x16)
# print(x16.shape)

# x17 = np.identity(3)
# print(x17)

# x18 = np.zeros((8,8))
# x18 = np.pad(x18, pad_width=1, mode='constant',constant_values=1)
# print(x18)
# x18 = np.zeros((10,10))
# x18[0::9,]=1
# x18[::,0::9]=1
# print(x18)

# x19 = np.zeros((5,5))
# for i in range(5):
#     x19[i,i]=i+1
# print(x19)

# x19 = np.zeros((5,5))
# x19 = np.diag([1,2,3,4,5])
# print(x19)

# x20 = np.zeros((4,4))
# x20[::2,1::2]=1
# x20[1::2,::2]=1
# print(x20)

# x21 = np.full(27,np.random.randn()).reshape(3,3,3)
# print(x21)
# x21 =np.random.random(3,3,3)
# print(x21)

# x22 = np.random.randint(1,10,(3,3))
# print(x22)
# print('sum of all :', x22.sum())
# print('sum of columns :', x22.sum(axis=0))
# print('sum of rows :', x22.sum(axis=1))

# x23 = np.array([1,2,3])
# x24 = np.array([4,5,6])
# x25 = np.inner(x23,x24)
# print(x25)

# x26 = np.random.randint(1,10,(3,3))
# print(x26)
# x27 = np.arange(0,3)
# print(x27)
# print(x26 + x27)

# lista = [1,2,3]
# array = np.array(lista)
# print(lista == array.tolist())

# x = np.arange(0,np.pi*5,0.2)
# y = np.sin(x)
#
# plt.plot(x,y)
# plt.show()

# def sum_matrix_Elements(m):
#     arra = np.array(m)
#     element_sum = 0
#     for p in range(len(arra)):
#         for q in range(len(arra[p])):
#             if arra[p][q] == 0 and p < len(arra)-1:
#                 arra[p+1][q] = 0
#             element_sum += arra[p][q]
#     return element_sum
# m = [[1, 1, 0, 2],
#           [0, 3, 0, 3],
#           [1, 0, 4, 4]]
# print("Original matrix:")
# print(m)
# print("Sum:")
# print(sum_matrix_Elements(m))

# nums = np.array([[3, 2, 1], [6, 5, 4], [np.nan, 8, 7]])
# print(nums[~np.any(np.isnan(nums), axis=1)])

# a = np.array([1, 2, 3, 3])
# print("a :", a)
# b = np.array([1, 2, 3, 3.0000001])
# print("b :", b)
# print(np.isclose(a, b, rtol=1e-15))
# np.set_printoptions(precision=15)
# print(a == b)

# nums = np.array([[5.54, 3.38, 7.99],
#               [3.54, 8.32, 6.99],
#               [1.54, 2.39, 9.29]])
# print("Original array:")
# print(nums)
# print("\nNew array of equal shape and data type of the said array filled by 0:")
# print(np.zeros_like(nums))

# nums = np.array([[5.54, 3.38, 7.99],
#               [3.54, 8.32, 6.99],
#               [1.54, 2.39, 9.29]])
# print("Original array:")
# print(nums)
# n = 8.32
# r = 18.32
# nums = np.where(nums==8.32,r, nums)
# print(nums)

# x = np.tril(np.arange(2, 14).reshape(4, 3))
# print(x)

#numpy exercices arrays

# fahrenheit_array = np.array([0, 12, 45.21, 34, 99.91, 32])
# celcius_array = np.round(5/9*(fahrenheit_array - 32),2)
# print(celcius_array)
# print(celcius_array.size)
# print(celcius_array.itemsize)
# print(celcius_array.nbytes)

# array_1 = [0,10,20,40,60]
# array_2 = [0,40]
# print(np.in1d(array_1,array_2))

# array_1 = np.arange(0,50)
# array_2 = np.array([10,30,40,60])
# print(np.intersect1d(array_2,array_1))

# array = [10,10,20,20,30,30]
# print(np.unique(array))

# array_1 = [10,20,40,60,80]
# array_2 = [10,30,40,50,70]
# print(np.setdiff1d(array_1, array_2))
# print(np.setxor1d(array_1, array_2))
# print(np.union1d(array_1, array_2))

# array_3 = [True, True, False, True]
# array_4 = [0,1,2,3]
# print(np.all(array_3)," ", np.all(array_4))

# array = [[True, False], [True, True]]
# print(np.all(array, axis=1))
# print(np.all(array, axis=0))

# array = np.arange(1,6)
# array = np.tile(array,3)
# print(array)
# array = np.arange(1,6)
# array = np.repeat(array,3)
# print(array)

# array_1 = np.array([1, 2])
# array_2 = np.array([4, 5])
# print(np.greater(array_1,array_2))
# print(np.greater_equal(array_1,array_2))
# print(np.less(array_1,array_2))
# print(np.less_equal(array_1,array_2))

# array_1 = np.array([[4, 6], [2, 1]])
# print(np.sort(array_1, axis=0))
# print(np.sort(array_1, axis=1))

# first_names = np.array(['Betsey', 'Shelley', 'Lanell', 'Genesis', 'Margery'])
# last_names = np.array(['Battle', 'Brien', 'Plotner', 'Stahl', 'Woolum'])
# print(np.lexsort((last_names, first_names)))

# array_1 = np.full((2, 3), 1)
# print(array_1)
# print(array_1.shape)
# print(array_1.dtype)
# array_2 = np.array(array_1,dtype='float64')
# print(array_2.dtype)
# array_3 = array_1.astype(float)
# print(array_3.dtype)

#exercise 41
# x = np.arange(4, dtype=np.int64)
# y = np.full_like(x, 10)
# print(y)












































