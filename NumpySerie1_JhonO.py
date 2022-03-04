#Numpy ejercicios serie 1
#canal youtube john oritz ordoñez

import numpy as np
tamaño = 3

d = np.empty(0, int)
d = np.full((tamaño, tamaño), 0)
f = np.empty(0, int)

for i in reversed(range(tamaño)):
    d[:, i] += i+1
    a = np.full((tamaño), i + 1)
    f = np.insert(f, 0, a)
f = np.append(f, d.flatten())

x = np.array([1, 2, 3])
print("\nfunciones repeat y tile para generar patrones\n")
print(f)
print(np.r_[np.repeat(x, 3), np.tile(x, 3)])

print("\nfuncion all para verificar si una matriz tiene un elemento igual a cero\n")
a = np.array([1, 3, 5])
b = np.array([1, 0, 2])
print("[1, 3, 5]", np.all(a), "\n[1, 0, 2]", np.all(b))

print("\nfuncion all para verificar si una matriz tiene un elemento distinto de cero\n")
a = np.array([1, 3, 5])
b = np.array([0, 0, 0])
print("[1, 3, 5]", np.any(a), "\n[0, 0, 0]", np.any(b))

print("Comprobar elemento a elemento la igualdad con un grado de tolerancia.\n")

a = np.array([1, 2, 3, 3])
print("a :", a)
b = np.array([1, 2, 3, 3.0000001])
print("b :", b)
print(np.isclose(a, b, rtol=1e-5))

print("Crear una matriz 4x3 e imprimir cada elemento\n")

arreglo = np.arange(1, 13).reshape((4, 3))
print(arreglo, '\n')

for elemento in np.nditer(arreglo):
    print(elemento, end=' ')

print("Crear un arreglo con los valores 0 a 9, y cambiar el signo de los valores 5 a 7.\n")

arreglo = np.arange(10)

arreglo[(arreglo >= 5) & (arreglo <= 7)] *= -1

print(arreglo)

print("Crear una matriz de 4x4 con 0 y 1 escalonados. "
      "Los valores de la diagional principal deben ser igual a 0.\n")

m = np.zeros((4, 4))

m[::2, 1::2] = 1
m[1::2, ::2] = 1

print(m)

print("Sumar un vector a cada fila de una matriz.\n")

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

a = np.array([1, 1, 0])

resultado = np.empty_like(m)

for i in range(4):
    resultado[i, :] = m[i, :] + a

print(resultado)

print("Crear un arreglo a partir de una lista, y luego obtener la lista original.\n")

lista = [2, 3, 5, 7, 11]

arreglo = np.array(lista)

lista2 = arreglo.tolist()

print(lista == lista2)

print("Creacion de gráfica en numpy")

import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.2)
y = np.sin(x)

plt.plot(x, y)
plt.show()

