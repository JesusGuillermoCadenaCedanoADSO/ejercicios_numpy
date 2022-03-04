import numpy as np

print("producto punto de dos matrices")

a = [[2, 3], [5, 7]]
print(a)
b = [[4, 2], [3, 5]]
print(b)
print(np.dot(a, b))

print("producto externo de dos vectores")

a = [[2, 2], [1, 2]]
print(a)
b = [[0, 1], [4, 0]]
print(b)
print(np.outer(a, b))

x = np.array(['a', 'b', 'c'], dtype=object)
print(x)
print("np.outer(x, [1, 2, 3])\n", np.outer(x, [1, 2, 3]))

print("producto cruz de dos vectores")

p = [[1, 3], [4, 3]]
print(p)
q = [[5, 9], [3, 2]]
print(q)
print(np.cross(p, q))

print("determinante de matriz")

m = np.array([[2, 3, 4], [5, 7, 8], [2, 3, 5]])
print(m)
print(np.linalg.det(m))

print("Computar el producto interno de dos vectores unidimensionales")

a = np.array([2, 3, 5])
print(a)
b = np.array([3, 8, 7])
print(b)
print(np.inner(a, b))


print("Computar la norma de un vector o una matriz.")

v = np.arange(1, 13)
print(v)
print(np.linalg.norm(v))

print("Computar el inverso de una matriz dada.")


m = np.array([[3, 2, 1], [7, 5, 3], [7, 4, 5]])

print(m)

print(np.linalg.inv(m))

print("Generar 10 números aleatorios a partir de una distribución normal.")

arreglo = np.random.normal(size=10)

print(arreglo)

print("Generar 10 valores enteros entre 1 y 30.")

print(np.random.randint(low=1, high=30, size=10))

print("Crear una arreglo con 10 elementos y luego aleatorizar sus posiciones.")

print(np.random.permutation(10))

print("normalizar una matriz")

v = np.random.rand(10)
minimo = v.min()
maximo = v.max()

normalized_v = (v-minimo)/(maximo-minimo)
print(normalized_v)

print("Encontrar el valor más cercano a un valor arbitrario.")

a = np.random.uniform(1, 20, 10)

valor = 5

print(a,"\n",valor,"\n",a.flat[np.abs(a - valor).argmin()])

print("Encontrar el valor más frecuente en un arreglo.")

a = np.random.randint(0, 15, 20)
print(a)
print(np.bincount(a).argmax())

print("A partir de una matriz con coordenadas cartesianas"
      " realizar la conversión a coordendas polares")

m = np.random.random((15, 2))

x = m[:,0]
y = m[:,1]

r = np.sqrt(x**2 + y**2)

t = np.arctan2(y, x)

print("obtener los n numeros mayores en un arreglo")

a = np.arange(15)
print(a)
np.random.shuffle(a)
print(a)
n = 3
print(a[np.argsort(a)[-n:]])

print("Ordenar una matriz por diferentes ejes, y por la versión arreglo de sus elementos.")

m = np.array([[3, 2, 1], [8, 4, 3], [5, 6, 0]])

print(m)
print("axis=0\n", np.sort(m, axis=0))
print("axis=1\n", np.sort(m, axis=1))
print("axis=None\n", np.sort(m, axis=None))


print("Crear una estructura de datos para representar a un estudiante,"
      " crear registros, y ordenar por el identificador de clase y nota.")

tipo_dato_estudiante = [('nombre', 'S15'), ('curso', int), ('nota', float)]

estudiantes = [('Edward', 1, 4.5), ('German', 2, 4.3), ('Daniela', 1, 4.4)]

arreglo_estudiantes = np.array(estudiantes, dtype=tipo_dato_estudiante)
print(arreglo_estudiantes)

print("ordenado\n", np.sort(arreglo_estudiantes, order=['curso', 'nota']))

print("Obtener e imprimir los índices que describen el ordendamiento por múltiples columnas.")

edades = np.array([27, 21, 29, 34, 37, 19, 23, 24, 18, 31,19])
alturas = np.array([1.72, 1.65, 1.68, 1.55, 1.63, 1.73, 1.81, 1.69, 1.75, 1.83,1.80])

indices = np.lexsort((edades, alturas))

for i in indices:
    print(edades[i], alturas[i])

print("Obtener los índices de ordenamiento de un arreglo.")

edades = np.array([13, 18, 1, 5, 8, 23, 27, 29, 31, 4])
print("edades\n", edades)

indices = np.argsort(edades)

print("indices\n", indices)

xyz = np.array([2, 0, 1, 4])
print("xyz\n", xyz)
print("indices\n", np.argsort(xyz))
print("xyz ordenado\n", xyz[np.argsort(xyz)])

print("Ordenar los primeros 10 elementos de un arreglo.")

arreglo = np.random.random(20)
print(arreglo)
print(arreglo[np.argpartition(arreglo, range(10))])

print("Ordenar los elementos a partir de una ubicación dada.")
a = np.array([70, 50, 20, 30, 100, 60, 50, 40])
print(a)
print(np.partition(a, 3))


