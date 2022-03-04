import numpy as np

print("Crear una matriz a partir de un arreglo que valores enteros arbitrarios.")

arreglo = np.arange(10, 22)

matriz = arreglo.reshape(3, 4)

print("arreglo\n", arreglo, "\nmatriz\n", matriz)

print("Crear un arreglo a partir de un rango.")
print("rango\n", range(1, 11))
arreglo2 = np.array(list(range(1, 11)))
print("\narreglo\n", arreglo2)

print("Convertir los elementos de un arreglo a real (punto flotante, o float).\n")

lista = [2, 3, 5, 7, 11]
arreglo = np.asfarray(lista)

print("lista\n",lista,"\narreglo\n",arreglo)

print("Crear un arreglo con elementos iguales a 1, "
      "y luego agregar un marco (o padding) con valores iguales a 0.")

unos = np.ones((3, 3), dtype=np.int)

print(unos, "\n", np.pad(unos, pad_width=1, mode='constant', constant_values=0))

print("Crear una matriz que siga el patrón de distribución de celdas (casillas) de un tablero de ajedrez.")

tablero = np.zeros((8, 8), dtype=int)
tablero[::2, 1::2] = 1
tablero[1::2, ::2] = 1

print("\nConvertir a grados Fahrenheit cada uno de los elementos "
      "de un arreglo que contiene medidas en grados centigrados.")

centigrados = np.array([0, 32.5, 42.8, 13.9])

fahrenheit = centigrados * 9/5 + 32

print(fahrenheit)

print("\nEncontrar los índices de los elementos menor y mayor en un arreglo con valores arbitrarios.")

arreglo = np.array([7, 13, 9, 0, 8, 19, 14, 21, 1, 29])
print(arreglo)
print("minimo:", np.argmin(arreglo))
print("maximo:", np.argmax(arreglo))

print("Ordenar un grupo de apellidos y nombres y obtener los índices del ordenamiento resultante.")

nombres = ['Edward', 'Angela', 'Daniela', 'German']
apellidos = ['Ortiz', 'Burgos', 'Meneses', 'Urbano']

print(np.sort((nombres, apellidos)))
print(np.lexsort((nombres, apellidos)))

print("Crear una matriz con valores iguales a uno a partir de una diagonal arbitraria.")

matriz = np.tri(5, 4, -1)

print(matriz)

print("Particionar un arreglo en cuatro (4) subarreglos.")

arreglo = np.arange(1, 20)
print(arreglo)
print(np.split(arreglo, [4, 10, 15]))

print("Comprobar si un valor arbitrario se encuentra de los valores almacenados en un arreglo.")

matriz = np.array([[1, 14, 17.5, 9], [9, 7, 4, 13], [1, 2, 0.5, 3]])
print("matriz\n", matriz)
print("7", 7 in matriz)
print("7.9", 7.9 in matriz)

print("Crear un arreglo con 20 valores uniformemente espaciados 0 y 1 (excluídos los extremos).\n")

arreglo = np.linspace(0, 1, 22, endpoint=True)

arreglo = arreglo[1:-1]

print("Especificar que un arreglo sea de sólo lectura (no modificable).")

arreglo = np.array([1, 2, 3, 4, 5])
arreglo.flags.writeable = False

print("Encontrar los elementos que son múltiplo de 2 y 3 en un arreglo; luego calcular su suma.")

arreglo = np.arange(1, 101)

arreglo = arreglo[(arreglo % 2 == 0) | (arreglo % 3 == 0)]

print(arreglo.sum())

print("Iterar una matriz siguiendo el mecanismo de recorrido basado en Fortran.")

matriz = np.arange(1, 13).reshape(4, 3)
print("matriz\n", matriz)
for elemento in np.nditer(matriz, order='F'):
    print(elemento, end=' ')

print("\nIterar una matriz y mutiplicar cada elemento por un valor arbitrario.")

matriz = np.arange(16).reshape(4, 4)

print("matriz\n", matriz)
for elemento in np.nditer(matriz, op_flags=['readwrite']):
    elemento[...] = 7 * elemento

print("matriz * 7\n", matriz)

print("Crear una matriz con columnas de diferente tipo de dato.\n")

matriz = np.zeros((3, ), dtype=('i4,f4,a64'))

datos = [(1, 4.1, 'Cronica Roja'), (2, 4.3, 'La virgen de los sicarios'), (3, 4.2, 'Satanas')]

print(datos,"\n")
matriz[:] = datos
print(matriz)


print("A través de una función elevar al cubo cada elemento de un arreglo.")

def elevar_cubo(arreglo):
      iterador = np.nditer([arreglo, None])

      for a, b in iterador:
            b[...] = a * a * a

      return iterador.operands[1]

print([3, 5, 7],"\n",elevar_cubo([3, 5, 7]))


print("Convertir un arreglo con valores reales en uno con valores enteros.")

arreglo = np.arange(1, 9, dtype=np.float)

print(arreglo, "\n")

arreglo = arreglo.astype(np.int)

print(arreglo, "\n")

print("Remover las filas que contengan valores no numéricos.")

m = np.array([[1, 2, 3, 4], [8, 9, 7, 3], [np.nan, np.nan, 8, 7]])

print(m, "\n")

m = m[~np.isnan(m).any(axis=1)]

print(m)

print("Obtener los elementos únicos de un arreglo, y además sus frecuencias.")

a = np.array([2, 3, 5, 3, 3, 5, 2, 7, 9, 5, 9])

unicos, conteos = np.unique(a, return_counts=True)

print(a, "\n")

print(unicos, conteos)

print("Mover elementos de un arreglo a otro por medio de índices.")

a = np.array([1, 1, 2, 3, 3, 0])
print("a", a)
b = np.array([0, 4, 60])
print("b", b)
a.put([0, 4, 5], b)

print("a.put([0, 4], b) :", a)