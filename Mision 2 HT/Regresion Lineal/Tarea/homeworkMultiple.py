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

#Seleccionamos las columna GrLivArea, BedroomAbvGr, LotArea
X_multiple = californiaCasas.frame[['GrLivArea', 'BedroomAbvGr', 'OverallQual']]

#Defino los datos correspondientes a las etiquetas
y_multiple = californiaCasas.target

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size=0.2)

#Defino el algoritmo a utilizar
lr_multiple = linear_model.LinearRegression()

#Entreno el modelo
lr_multiple.fit(X_train, y_train)

#Realizo una predicción
Y_pred_multiple = lr_multiple.predict(X_test)

print('DATOS DEL MODELO REGRESIÓN LINEAL MULTIPLE')

print('Valor de las pendientes o coeficientes "a":')
print(lr_multiple.coef_)

print('Valor de la intersección o coeficiente "b":')
print(lr_multiple.intercept_)

print('Precisión del modelo:')
print(lr_multiple.score(X_train, y_train))


