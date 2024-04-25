import numpy as np
import math

x1 = 1
x2 = 4
y1 = 2
y2 = 6

a = [1, 2, 3]
b = [4, 5, 6]

#Distancia Manhattan

def distancia_manhattan(x1, x2, y1, y2):
    distancia = abs(x2 - x1) + abs(y2 - y1)
    return distancia

d = distancia_manhattan(x1, x2, y1, y2)

print(f"La distancia de Manhattan entre los puntos ({x1}, {y1}) y ({x2}, {y2}) es: {d}")

#Distancia Chebyshev

def distancia_chebyshev(x1, y1, x2, y2):
    dist_cheby = max(abs(x1 - x2), abs(y1 - y2))
    return dist_cheby

d_ch = distancia_chebyshev(x1, x2, y1, y2)

print(f"La distancia de Chebyshev entre los puntos ({x1}, {y1}) y ({x2}, {y2}) es: {d_ch}")

#Distancia Canberra

def distancia_canberra(x_c, y_c):
    if len(x_c) != len(y_c):
        raise ValueError("Los vectores deben tener la misma longitud")
    
    distancia = 0
    for i in range(len(x_c)):
        distancia = distancia + abs(x_c[i] - y_c[i]) / (abs(x_c[i]) + abs(y_c[i]))
    return distancia

resultado = distancia_canberra(a, b)

print(f"La distancia de Canberra entre los vectores {a} y {b} es: {resultado:.2f}")

#Distancia de cosenos

def distancia_coseno(x_c, y_c):
    if len(x_c) != len(y_c):
        raise ValueError("Los vectores deben tener la misma longitud")
    
    producto_punto = np.dot(x_c, y_c)
    magnitud_x_c = np.linalg.norm(x_c)
    magnitud_y_c = np.linalg.norm(y_c)
    
    return producto_punto / (magnitud_x_c * magnitud_y_c)

resultado = distancia_coseno(a, b)

print(f"La distancia del Coseno entre los vectores {a} y {b} es: {resultado:.2f}")


