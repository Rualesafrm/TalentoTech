import numpy as np
import matplotlib.pyplot as plt

# Función para calcular la distancia Manhattan
def distancia_manhattan(x1, x2, y1, y2):
    distancia = abs(x2 - x1) + abs(y2 - y1)
    return distancia

# Calcular los promedios para cada especie
promedios_setosa = np.mean(iris_features[iris_target == 0], axis=0)
promedios_versicolor = np.mean(iris_features[iris_target == 1], axis=0)
promedios_virginica = np.mean(iris_features[iris_target == 2], axis=0)

# Calcular las distancias Manhattan entre los promedios de las especies
distancia_setosa_versicolor = distancia_manhattan(promedios_setosa[0], promedios_setosa[1], promedios_versicolor[0], promedios_versicolor[1])
distancia_setosa_virginica = distancia_manhattan(promedios_setosa[0], promedios_setosa[1], promedios_virginica[0], promedios_virginica[1])
distancia_versicolor_virginica = distancia_manhattan(promedios_versicolor[0], promedios_versicolor[1], promedios_virginica[0], promedios_virginica[1])

# Mostrar las distancias
print("Distancia Setosa-Versicolor:", distancia_setosa_versicolor)
print("Distancia Setosa-Virginica:", distancia_setosa_virginica)
print("Distancia Versicolor-Virginica:", distancia_versicolor_virginica)

# Graficar las distancias
etiquetas = ['Setosa-Versicolor', 'Setosa-Virginica', 'Versicolor-Virginica']
distancias = [distancia_setosa_versicolor, distancia_setosa_virginica, distancia_versicolor_virginica]

plt.bar(etiquetas, distancias, color=['blue', 'green', 'red'])
plt.xlabel('Combinaciones de especies')
plt.ylabel('Distancia de Manhattan')
plt.title('Distancias de Manhattan entre promedios de especies de Iris')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejorar la legibilidad
plt.show()
