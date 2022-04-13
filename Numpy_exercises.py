# https://www.w3resource.com/python-exercises/numpy
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt

# Basic exercises

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

# numpy exercices arrays

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

# exercise 41
# x = np.arange(4, dtype=np.int64)
# y = np.full_like(x, 10)
# print(y)


# exercise 43
# x = np.diagflat([4,5,6])
# print(x)

# exercise 45
# x = np.linspace(2.5, 6.5, 30)
# print(x)

# exercise 48
# x = np.triu(np.arange(2,14).reshape(4,3),-1)
# print(x)

# exercise 50
# x = np.arange(2,10).reshape(2,4)
# e1 = x.flat[3]
# print(x,'\n',e1)

# exercise 52
# x = np.zeros((2,3,4))
# print(x)
# print(np.moveaxis(x,0,-1).shape)
# print(np.moveaxis(x,0,-1))

# exercise 53
# x = np.ones((2,3))
# print(np.rollaxis(x,1,0))

# exercise 54
# x = np.zeros((3, 4))
# y = np.expand_dims(x, axis=1)
# print(y)

# exercise 57
# x = np.zeros((1, 2, 1))
# print(x,'\n',np.squeeze(x),'\n',np.squeeze(x).shape)

# exercise 61
# x = np.arange(1, 15)
# print("Original array:",x)
# print("After splitting:")
# print(np.split(x, [2, 9]))

# exercise 62
# x = np.arange(16).reshape((4, 4))
# print("Original array:",x)
# print("After splitting horizontally:")
# print(np.hsplit(x,[1,2]))

# exercise 64
# ar = np.arange(5)
# ar2 = np.tile(ar,5).reshape(5,5)
# print(ar2)

# exercise 67
# x = np.zeros(10)
# x.flags.writeable = False

# exercise 73
# x = np.arange(12).reshape(3, 4)
# x = x*3
# print(x)

# exercise 75
# x = np.zeros((3,), dtype=('i4,f4,a40'))
# new_data = [(1, 2. , "Albert Einstein")]
# x[:] = new_data
# print(x)

# exercise 77
# x = np.arange(12).reshape(3, 4)
# y = np.transpose(x)
# print(y)

# exercise 78
# a1=np.array([1,2,3,4])
# a2=np.array(['Red','Green','White','Orange'])
# a3=np.array([12.20,15,20,40])
# result=np.core.records.fromarrays([a1,a2,a3],names='a,b,c')
# print(result)

# exercise 79
# x = np.arange(9).reshape(3,3)
# print(x,'\n','first column:\n',x[:,0])

# exercise 85
# iterable = (x for x in range(10))
# y=(np.fromiter(iterable, np.int))
# print(y)

# exercise 86
# x= np.array([[10,20,30],[40,50,60]])
# y= np.array([[100],[200]])
# x =np.append(x,y,axis=1)
# print(x)

# exercise 87
# x = np.array([[1,1],[1,1],[2,2],[3,3]])
# y = np.unique(x)
# z = np.unique(x,axis=0)
# print(y)
# print(z)

# exercise 88
# array = np.array([[0.42436315, 0.48558583, 0.32924763],
# [0.7439979, 0.58220701, 0.38213418],
# [0.5097581, 0.34528799, 0.1563123]])
# array[array>0.5]=0.5
# print(array)

# exercise 89
# ar=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# print(np.delete(ar,[0,3,4]))

# exercise 90
# ar = np.array([[1., 2., 3.],
#                [4., 5., np.nan],
#                [7., 8., 9.],
#                [1., 0., 1.]])
#
# ar2 = ~np.isnan(ar)
# ar3 = np.all(ar2, axis=1)
# print(ar2)
# print(ar3)
# print(ar[ar3])

# exercise 92
# a = np.array([97, 101, 105, 111, 117])
# b = np.array(['a','e','i','o','u'])
# c = b[(a>100) & (a<110)]
# print(c)

# exercise 94
# a = np.array([10,10,20,10,20,20,20,30, 30,50,40,40])
# unique_elements, counts_elements = np.unique(a, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))


# exercise 96
# x = np.array([[20,20,20],[30,30,30],[40,40,40]])
# y = np.array([20,30,40])
# z = np.reshape(y,(3,1))
# print(x/z)
# print(x/y[:,None])


# exercise 99
# ar = np.array([10,20,30])
# print(np.sum(ar))
# print(ar.prod())

# exercise 100
# ar = np.array([10,10,20,30,30])
# ar[0] = 0
# ar[4] = 40
# print(ar)

# exercise 104
# ar = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(ar[:,-2:])


# exercise 110
# ar = np.array([200, 300, np.nan, np.nan, np.nan, 700])
# ar2 = ~np.isnan(ar)
# print(ar[ar2])
# ar3 = np.array([[1,2,3],[np.nan, 0, np.nan],[6,7,np.nan]])
# ar4 = ~np.isnan(ar3)
# print(ar3[ar4])

# exercise 118
# n= 4
# nums = np.arange(-6, 6)
# print("\nOriginal array:")
# print(nums)
# print("\nPosition of the index:")
# print(np.argmax(nums>1))

# exercise 119
# arr = np.empty((0,3), int)
# arr = np.append(arr,np.array([[10,20,30]]),axis=0)
# arr = np.append(arr,np.array([[40,50,60]]),axis=0)
# print(arr)

# exercise 120
# ar = np.array([[1,2,3],[4,8,1]])
# i,j = np.unravel_index(ar.argmax(),ar.shape)
# print(ar[i,j])

# exercise 122
# x = np.reshape(np.arange(16),(4,4))
# result = x[[0,1,2],[0,1,3]]
# print(x,'\nresult:',result)

# exercise 149
# a = np.array([1,3,7,9,10,13,14,17,29])
# result = np.where(np.logical_and(a>=7, a<=20))
# print(result)

# exercise 150
# ar = np.arange(12).reshape(3, 4)
# ar[:,[1,0]]=ar[:,[0,1]]
# print(ar)

# exercise 151
# ar = np.arange(36).reshape(4,9)
# result = np.where(np.any(ar>10, axis=1))
# print(ar,'\n',result)

# exercise 152
# ar = np.arange(36).reshape(4,9)
# print(ar.sum(axis=0))

# exercise 153
# num = np.arange(18)
# arr1 = np.reshape(num, [6, 3])
# result = arr1[np.triu_indices(3)]
# print(result)

# exercise 154
# result  = np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
# print("\nCopy of a matrix with the elements below the k-th diagonal zeroed:")
# print(result)

# exercise 156
# arr1 = np.array([[10, 20 ,30], [40, 50, np.nan],
#                  [np.nan, 6, np.nan], [np.nan, np.nan, np.nan]])
# print("Original array:")
# print(arr1)
# temp = np.ma.masked_array(arr1, np.isnan(arr1))
# result = np.mean(temp,axis=1)
# print(np.mean([10,20,30,40,50,6]))
# print(result.filled(np.nan))

# exercise 157
# ar = np.array([1, 2, 3, 2, 4, 6, 1, 2, 12, 0, -12, 6])
# lista = []
# for i in range(2, len(ar), 3):
#     lista.append(np.mean(ar[i - 2:i+1]))
# ar2 = np.array(lista)
# print(ar2)
# print(ar.reshape(-1, 3))
# print(np.mean(ar.reshape(-1, 3), axis=1))

# exercise 158
# ar1 = np.array([[0, 1], [2, 3]])
# ar2 = np.array([[4, 5], [0, 3]])
# mean = (ar1 + ar2)/2
# print(mean)

# exercise 159
# ar = np.array([[11, 22, 33, 44, 55],
#                [66, 77, 88, 99, 100]])
# i = [1, 3, 0, 4, 2]
# result = ar[:, i]
# print(result)

# exercise 160
# ar = np.array([1, 7, 8, 2, 0.1, 3, 15, 2.5])
# k = 4
# lista = []
# for i in range(k):
#     menor = np.min(ar)
#     indice = np.where(ar == menor)
#     if menor not in lista:
#         lista.append(menor)
#     ar = np.delete(ar, [indice])
# print(lista)
# ar = np.array([1, 7, 8, 2, 0.1, 3, 15, 2.5])
# result = np.argpartition(ar, k)
# print(ar[result[:k]])


# exercise 162
# a = np.random.randint(0, 10, (3, 4, 8))
# print("Original array and shape:")
# print(a)
# print(a.shape)
# print("--------------------------------")
# tidx = np.random.randint(0, 3, 4)
# print("tidex: ",tidx,len(tidx))
# print("Result:")
# print(a[tidx, np.arange(len(tidx)),:])

# exercise 163
# x = np.array([10, -10, 10, -10, -10, 10])
# y = np.array([.85, .45, .9, .8, .12, .6])
# print("Original arrays:")
# print(x)
# print(y)
# result = np.sum((x == 10) & (y >= .9))
# print("\nNumber of instances of a value occurring in"
#       " one array on the condition of another array:")
# print(result)

# exercise 166
# ar1 = np.array(['PHP', 'JS', 'C++', 'z'])
# ar2 = np.array(['Python', 'C#', 'NumPy', 'x', 'v'])
# result = ar1.tolist() + ar2.tolist()
# result = result[:len(ar1) - 1] + \
#          [''.join(result[len(ar1) - 1]
#                   + result[len(ar1)])] + result[len(ar1)+1:]
# print(np.array(result))
# result = np.r_[ar1[:-1], [ar1[-1]+ar2[0]], ar2[1:]]
# print(result)

# exercise 167: Convert a Python dictionary to a Numpy ndarray
# dict_created = {0: 0, 1: 1, 2: 8, 3: 27,
#                 4: 64, 5: 125, 6: 216}
# print(type(dict_created))
# print(np.array(list(dict_created.items())))
# print(np.array(list(dict_created.values())))
#
# udict = """{"column0":{"a":1,"b":0.0,"c":0.0,"d":2.0},
#    "column1":{"a":3.0,"b":1,"c":0.0,"d":-1.0},
#    "column2":{"a":4,"b":1,"c":5.0,"d":-1.0},
#    "column3":{"a":3.0,"b":-1.0,"c":-1.0,"d":-1.0}
#   }"""
# t = literal_eval(udict)
# print("\nOriginal dictionary:")
# print(t)
# print("Type: ",type(t))
# result_nparra = np.array([[v[j] for j in ['a', 'b', 'c', 'd']] for k, v in t.items()])
# print("\nndarray:")
# print(result_nparra)
# print("Type: ",type(result_nparra))


# exercise 168
# np_array = np.random.rand(12,3,2)
# print("Original Numpy array:")
# print(np_array)
# print("Type: ",type(np_array))
# df = pd.DataFrame(np.random.rand(12,3),columns=['A','B','C'])
# print("\nPanda's DataFrame: ")
# print(df)
# print("Type: ",type(df))


# exercise 169
# np_array = np.arange(3*4*5).reshape(3,4,5)
# print("Original Numpy array:")
# print(np_array)
# print("Type: ",type(np_array))
# result = np.diagonal(np_array, axis1=1, axis2=2)
# print("\n2D diagonals: ")
# print(result)
# print("Type: ",type(result))

# exercise 170
# np_array = np.array([[1, 2, 3], [2, 1, 2]], np.int32)
# print("Original Numpy array:")
# print(np_array)
# print("Type: ",type(np_array))
# print("Sequence: 1,2",)
# result = repr(np_array).count("1, 2")
# print("Number of occurrences of the said sequence:",result)

# exercise 171
# np_array = np.array([[1,2,3], [4,5,6] , [7,8,9], [10, 11, 12]])
# test_array = np.array([4,5,6])
# print("Original Numpy array:")
# print(np_array)
# print("Searched array:")
# print(test_array)
# print("Index of the searched array in the original array:")
# print(np.where((np_array == test_array).all(1))[0])

# exercise 172
# arra = np.array([[1,  1,  0],
#                  [0,  0,  0],
#                  [0,  2,  3],
#                  [0,  0,  0],
#                  [0, -1,  1],
#                  [0,  0,  0]])
#
# print("Original array:")
# print(arra)
# temp = {(0, 0, 0)}
# result = []
# for idx, row in enumerate(map(tuple, arra)):
#     if row not in temp:
#         result.append(idx)
# print("\nNon-zero unique rows:")
# print(arra[result])

