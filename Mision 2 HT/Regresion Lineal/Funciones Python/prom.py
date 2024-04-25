import numpy as np
import matplotlib.pyplot as plt

# Función para calcular la distancia Manhattan
def distancia_manhattan(x1, x2, y1, y2):
    distancia = abs(x2 - x1) + abs(y2 - y1)
    return distancia

# Cargar el conjunto de datos Iris
from sklearn.datasets import load_iris
iris_data = load_iris()
iris_features = iris_data.data  # Características (atributos)
iris_target = iris_data.target  # Variable objetivo (clase)

# Calcular los promedios para cada especie
promedios_setosa = np.mean(iris_features[iris_target == 0], axis=0)
promedios_versicolor = np.mean(iris_features[iris_target == 1], axis=0)
promedios_virginica = np.mean(iris_features[iris_target == 2], axis=0)

# Calcular las distancias Manhattan entre los promedios de las especies
distancia_setosa_versicolor = distancia_manhattan(promedios_setosa[0], promedios_setosa[1], promedios_versicolor[0], promedios_versicolor[1])
distancia_setosa_virginica = distancia_manhattan(promedios_setosa[0], promedios_setosa[1], promedios_virginica[0], promedios_virginica[1])
distancia_versicolor_virginica = distancia_manhattan(promedios_versicolor[0], promedios_versicolor[1], promedios_virginica[0], promedios_virginica[1])


# Graficar las distancias
nombres_especies = ['Setosa-Versicolor', 'Setosa-Virginica', 'Versicolor-Virginica']
distancias = [distancia_setosa_versicolor, distancia_setosa_virginica, distancia_versicolor_virginica]

plt.figure(figsize=(8, 6))
plt.bar(nombres_especies, distancias, color=['blue', 'green', 'orange'])
plt.title('Distancias Manhattan entre promedios de especies')
plt.xlabel('Pares de especies')
plt.ylabel('Distancia Manhattan')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
