from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(x1, x2):
    return np.sqrt(np.sum((x2 - x1) ** 2))

# Cargar el conjunto de datos Iris
iris_data = load_iris()
iris_features = iris_data.data  # Características (atributos)
iris_target = iris_data.target  # Variable objetivo (clase)

# Calcular los promedios para cada especie
promedios_setosa = np.mean(iris_features[iris_target == 0], axis=0)
promedios_versicolor = np.mean(iris_features[iris_target == 1], axis=0)
promedios_virginica = np.mean(iris_features[iris_target == 2], axis=0)

# Calcular las distancias euclidianas entre cada par de promedios
distancia_setosa_versicolor = distancia_euclidiana(promedios_setosa, promedios_versicolor)
distancia_setosa_virginica = distancia_euclidiana(promedios_setosa, promedios_virginica)
distancia_versicolor_virginica = distancia_euclidiana(promedios_versicolor, promedios_virginica)

# Graficar las distancias
distancias = [distancia_setosa_versicolor, distancia_setosa_virginica, distancia_versicolor_virginica]
nombres = ['Setosa - Versicolor', 'Setosa - Virginica', 'Versicolor - Virginica']

plt.bar(nombres, distancias, color=['blue', 'green', 'red'])
plt.xlabel('Pares de especies')
plt.ylabel('Distancia euclidiana')
plt.title('Distancias euclidianas entre promedios de especies en el dataset Iris')
plt.show()
