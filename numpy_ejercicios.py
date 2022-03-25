import numpy as np
# tamaño = 3
#
# x = np.array([1, 2, 3])
#
# ar1 = np.empty(0, int)
# ar2 = np.zeros((3, 3), int)
# ar3 = np.empty(0, int)
#
# for i in reversed((range(tamaño))):
#     ar2[:, i] += i+1
#     ar1 = np.full(tamaño, i+1)
#     ar3 = np.insert(ar3,0,ar1)
#
# ar4 = np.append(ar3, ar2.flatten())
#
# print(ar1)
# print(ar2)
# print(ar3)
# print(ar4)
#
# ar5 = np.repeat(x,3)
# ar6 = np.tile(x,3)
# ar7 = np.array([ar5,ar6])
# ar8 = np.r_[ar7]
# for i in np.nditer(ar8):
#     print(i, end = ' ')

# ar = np.arange(10)
#
# print(ar)
#
# ar[(ar>=5) & (ar<=7)] *= -1
#
# print(ar)

# ar = np.zeros((4, 4), int)
#
# print(ar)
#
# ar[::2,1::2]=1
# ar[1::2,::2]=1
#
#
# print(ar)
#
# arreglo = np.random.rand(10)*10
# arreglo = np.round(arreglo, decimals=2)
#
#
# print(arreglo)
# matriz = np.zeros((8,8), int)
#
# matriz[::2,1::2]=1
# matriz[1::2,::2]=1

# arreglo = np.arange(1, 20)
# arreglo=(arreglo[(arreglo%2==0) | (arreglo%3==0)])
# print(np.sum(arreglo))

# m = np.array([[1, 2, 3, 4], [8, 9, 7, 3], [np.nan, np.nan, 8, 7]])
#
# print(m[~np.isnan(m).any(axis=1)])

# a = np.array([1, 1, 2, 3, 3, 0])
# print("a", a)
# b = np.array([0, 4, 60])
# print("b", b)
# a.put([0, 4, 5], b)
# print(a)

# a = [[2, 3], [5, 7]]
# print(a)
# b = [[4, 2], [3, 5]]
# print(b)
# print(np.dot(a, b))

# a = np.random.randint(1, 10, 5)
# print(a)
# print(np.bincount(a))

# b = np.random.randint(0, 10, 5)
# np.random.shuffle(b)
# print(np.sort(b),'\n',np.sort(b)[-3:])

# edades = np.array([27, 21, 29, 34, 37, 19, 23, 24, 18, 31,19])
# alturas = np.array([1.72, 1.65, 1.68, 1.55, 1.63, 1.73, 1.81, 1.69, 1.75, 1.83,1.80])
#
# indices = np.lexsort((edades, alturas))
# print(indices)

# a7 = np.array([[1, 2, 3], [4, 5, 6]])
#
# print(np.delete(a7, 1))
# print(np.delete(a7, 3))
# print(np.delete(a7, 4))
# print(np.delete(a7, 1, 0))
# print(np.delete(a7, 1, 1))
# print(np.delete(a7, 0, 0))
# print(np.delete(a7, 0, 1))
# print(np.delete(a7, -1, 1))

# a8 = np.array([[1, 2, 3, 4, 5],
#                [6, 7, 8, 9, 10],
#                [11, 12, 13, 14, 15],
#                [16, 17, 18, 19, 20]])
#
# print(a8.size)
# a8.resize(4,2)
# print(a8)
# print(a8.size)

# arrays = np.array([np.random.randint(3, 6,(3,4)) for _ in range(2)])
# print(arrays)
# print(arrays.ndim)
#
# print(np.stack(arrays,axis=2))

# a9 = np.array([[1, 2, 3, 4, 5, 5.5],
#                [6, 7, 8, 9, 10, 10.5],
#                [11, 12, 13, 14, 15, 15.5],
#                [16, 17, 18, 19, 20, 20.5]])
#
# print()
# print(np.hsplit(a9, 2))

