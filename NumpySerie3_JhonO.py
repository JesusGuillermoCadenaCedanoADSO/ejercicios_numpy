import numpy as np

print("mostrar todas las fechas del mes de febrero de 2019\n")

print(np.arange('2019-02', '2019-03', dtype='datetime64[D]'))

print("Sumar un rango con 15 valores a un arreglo de fechas que contienela fecha 2019-09-12")

fechas = np.array('2019-09-12', dtype=np.datetime64)

print(fechas + np.arange(15))

print("Obtener la fecha de ayer, hoy, y mañana.")

ayer = np.datetime64('today', 'D') - np.timedelta64(1, 'D')

hoy = np.datetime64('today', 'D')

maghniana = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

print("ayer :", ayer, "hoy :", hoy, "maghniana :", maghniana)


print("Contar el número de días de febrero y marzo de 2019.")

print("febrero :", np.datetime64('2019-03-01') - np.datetime64('2019-02-01'))

print("marzo :", np.datetime64('2019-04-01') - np.datetime64('2019-03-01'))

