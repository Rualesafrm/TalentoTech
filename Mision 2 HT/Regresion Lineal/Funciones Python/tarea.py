from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris_data = load_iris()

def distancia_manhattan(x1, x2, y1, y2):
        distancia = abs(x2 - x1) + abs(y2 - y1)
        return distancia

    # Cargar el conjunto de datos Iris
iris_features = iris_data.data  # Características (atributos)
iris_target = iris_data.target  # Variable objetivo (clase)

    # Seleccionar dos puntos aleatorios
indices_aleatorios = np.random.choice(len(iris_features), 2, replace=False)
punto1 = iris_features[indices_aleatorios[0]]
punto2 = iris_features[indices_aleatorios[1]]

    # Extraer las características (atributos) de los puntos
x1, y1 = punto1[:2]  # Tomamos las dos primeras características (atributos)
x2, y2 = punto2[:2]  # Tomamos las dos primeras características (atributos)

    # Calcular la distancia de Manhattan entre los dos puntos
distancia = distancia_manhattan(x1, x2, y1, y2)

    # Imprimir el resultado
print(f"La distancia de Manhattan entre los puntos ({x1}, {y1}) y ({x2}, {y2}) es: {distancia}")



# Graficar los puntos
plt.scatter(x1, y1, color='blue', label='Punto 1')
plt.scatter(x2, y2, color='red', label='Punto 2')

# Dibujar la línea que representa la distancia de Manhattan
plt.plot([x1, x2], [y1, y1], color='green', linestyle='--')
plt.plot([x2, x2], [y1, y2], color='green', linestyle='--')

# Etiquetas de los ejes y leyenda
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('Ancho del sépalo (cm)')
plt.title('Distancia de Manhattan entre dos puntos')
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()






