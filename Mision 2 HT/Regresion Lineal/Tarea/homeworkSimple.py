#Se importan la librerias a utilizar
from sklearn import datasets, linear_model
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
californiaCasas = fetch_openml(name="house_prices", as_frame=True)

#Miro quien es mi target que en este caso es "SalePrice" que contiene los registros del precio de casas en California
print(californiaCasas.target)

#Verifico la información de las columnas
print('Nombres columnas:')
print(californiaCasas.feature_names)

#Seleccionamos "GrLivArea" que significa el área habitable por encima del nivel del suelo. 
X_simple = californiaCasas.frame['GrLivArea']

#Seleccionamos la columna 'SalePrice' como nuestra variable objetivo o target 
y_simple = californiaCasas.target

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2)

#Definimos el algoritmo a utilizar
lr_simple = linear_model.LinearRegression()

#Entrenamos el modelo
lr_simple.fit(X_train.values.reshape(-1, 1), y_train)

#Realizamos una predicción
y_pred_simple = lr_simple.predict(X_test.values.reshape(-1, 1))

# Imprimir los coeficientes y la precisión del modelo
print('Coeficiente:', lr_simple.coef_)
print('Intercepto:', lr_simple.intercept_)
print('Precisión del modelo:', lr_simple.score(X_train.values.reshape(-1, 1), y_train))
print(f'y = m * x + b = {lr_simple.coef_} * [GrLivArea] + {lr_simple.intercept_}')

# Graficar los datos de entrenamiento
plt.scatter(X_train, y_train, color='blue', label='Datos de entrenamiento')

# Graficar la recta de regresión
plt.plot(X_test, y_pred_simple, color='red', linewidth=2, label='Recta de regresión')

# Etiquetas y título
plt.xlabel('Área habitable (pies cuadrados)')
plt.ylabel('Precio de venta')
plt.title('Regresión lineal simple: Área habitable vs Precio de venta')

# Mostrar leyenda
plt.legend()

# Mostrar la gráfica
plt.show()


