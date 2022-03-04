import numpy as np
print("Sumar, restar, multiplicar y dividir escalares.")
print("np.add(1,3)", np.add(1, 3))
print("np.subtract(1,3)", np.subtract(1, 3))
print("np.multiply(1,3)", np.multiply(1, 3))
print("np.divide(1,3)", np.divide(1, 3))

print("Calcular el residuo al dividir los elementos (uno a uno) de un arreglo por un número arbitrario.")
a = np.arange(5)
print(a)
print("np.remainder(a, 3)", np.remainder(a, 3))

print("Redondear cada elemento de un arreglo a una cantidad de decimales arbitraria.")

arreglo = np.array([3.141592, 2.718281, 1.41421])
print(arreglo)
print("np.round(arreglo, decimals=5)", np.round(arreglo, decimals=5))

print("Rodendar los elementos de un arreglo al entero más cercano.")
a = np.array([-1.3, 2.7, 3.8, 3.3, 5.1, -4.7])
print("np.rint(a)", np.rint(a))

print("redondear hacia abajo")
print("np.floor(a)", np.floor(a))

print("redondear hacia arriba")
print("np.ceil(a)", np.ceil(a))

print("truncar")
print("np.trunc(a)", np.trunc(a))

print("multiplicar dos arreglos que contienen numeros complejos")

x = np.array([3+5j, 7+4j])
print("x",x)
y = np.array([2+3j, 8+2j])
print("y",y)

print("np.vdot(x, y)", np.vdot(x, y))


print("matriz por arreglo")

matriz = np.arange(15).reshape((3, 5))
print(matriz)

arreglo = np.arange(5)
print(arreglo)

print("np.inner(matriz, arreglo)", np.inner(matriz, arreglo))


print("encontrar las raices del polinomio: x^2 + 6x + 9")

print(np.roots([1, 6, 9]))

print("encontrar el valor del polinomio cuando x=2")

print(np.polyval([1, 6, 9], 2))

print("encontrar las operaciones de suma, resta, multiplicacion y division de polinomios")

print("a: 5x^2 + 10x + 15\n", "b: 15x^2+20x+25\n")

coef_a = (5, 10, 15)
coef_b = (15, 20, 25)

print("suma \n", np.polynomial.polynomial.polyadd(coef_a, coef_b))
print("resta \n", np.polynomial.polynomial.polysub(coef_a, coef_b))
print("multiplicar \n", np.polynomial.polynomial.polymul(coef_a, coef_b))
print("dividir \n", np.polynomial.polynomial.polydiv(coef_a, coef_b))






