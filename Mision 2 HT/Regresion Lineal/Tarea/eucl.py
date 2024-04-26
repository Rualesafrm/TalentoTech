from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

def distancia_euclidiana(x1, x2):
    return sum((x2 - x1) ** 2) ** 0.5

# Cargar el conjunto de datos Iris
iris_data = load_iris()
iris_features = iris_data.data  # Características (atributos)

# Seleccionar dos puntos 
punto1 = iris_features[0]
punto2 = iris_features[1]

# Calcular la distancia euclidiana entre los dos puntos
distancia = distancia_euclidiana(punto1, punto2)

# Imprimir el resultado
print("Distancia euclidiana entre los puntos:", distancia)

# Seleccionar dos características para graficar
x_index = 0  # Índice de la primera característica
y_index = 1  # Índice de la segunda característica

# Obtener las características de los dos puntos seleccionados
x1, y1 = punto1[x_index], punto1[y_index]
x2, y2 = punto2[x_index], punto2[y_index]

# Graficar los puntos
plt.scatter(x1, y1, color='blue', label='Punto 1')
plt.scatter(x2, y2, color='red', label='Punto 2')

# Dibujar una línea que conecta los puntos
plt.plot([x1, x2], [y1, y2], color='green', linestyle='--')

# Etiquetas de los ejes y leyenda
plt.xlabel('Característica {}'.format(x_index))
plt.ylabel('Característica {}'.format(y_index))
plt.title('Distancia euclidiana entre dos puntos en el dataset Iris')
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()
