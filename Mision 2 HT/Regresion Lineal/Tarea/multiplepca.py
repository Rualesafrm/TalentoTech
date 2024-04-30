# Se importan las librerías necesarias
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Se define la función para calcular el error cuadrático medio (MSE)
def calcular_rmse(y_real, y_pred):
# Y_real y Y_pred son listas
    n = len(y_real)
    mse = sum((y_real[i] - y_pred[i]) ** 2 for i in range(n)) / n
    rmse = mse**0.5
# Tambien se puede trabajar con numpy(es mas eficiente computacionalmente)
    return rmse

# Se carga el conjunto de datos de precios de casas en California
californiaCasas = fetch_openml(name="house_prices", as_frame=True)

# Seleccionamos las columnas de características
X_multiple = californiaCasas.frame[['GrLivArea', 'BedroomAbvGr', 'OverallQual']]

# Aplicamos PCA para reducir la dimensionalidad a dos componentes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_multiple)

# Visualizamos la varianza explicada por cada componente principal
print("Varianza explicada por cada componente principal:")
print(pca.explained_variance_ratio_)

# Definimos los datos correspondientes
y_multiple = californiaCasas.target

# Separación de los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_multiple, test_size=0.2)

# Definimos el modelo de regresión lineal múltiple
lr_multiple = LinearRegression()

# Entrenamos el modelo
lr_multiple.fit(X_train, y_train)

# Realizamos una predicción
Y_pred_multiple = lr_multiple.predict(X_test)

# Calculamos el error cuadrático medio
rmse = calcular_rmse(y_test.values, Y_pred_multiple)
print("Raiz cuadrada del error cuadrático medio:", rmse)

# Calculamos el coeficiente de determinación (R^2)
r2 = lr_multiple.score(X_test, y_test)
print("Coeficiente de determinación (R^2):", r2)

# Mostramos los coeficientes y la precisión del modelo
print('DATOS DEL MODELO REGRESIÓN LINEAL MÚLTIPLE')
print('Coeficientes:')
print(lr_multiple.coef_)
print('Intercepto:')
print(lr_multiple.intercept_)
print('Precisión del modelo:')
print(lr_multiple.score(X_train, y_train))

# Graficamos las predicciones y la línea de regresión pendiente graficar en 3 dimensiones
plt.scatter(y_test, Y_pred_multiple, color='blue', label='Predicciones')
plt.plot(y_test, y_test, color='red', linewidth=2, label='Línea de regresión')
plt.xlabel("Precios reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs Precios reales")
plt.legend()
plt.show()
