import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Datos
datos = {'X': [1, 2, 2, 3, 4, 4, 5, 6],
         'Y': [2, 3, 4, 4, 4, 6, 5, 7]}
df = pd.DataFrame(datos)
n = len(datos['X'])
df['X*Y'] = df['X'] * df['Y']
df['X^2'] = df['X'] ** 2

print(df)

# Calcular el promedio de X e Y
sum_X = df['X'].sum()
sum_Y = df['Y'].sum()

prom_X = df['X'].mean()
prom_Y = df['Y'].mean()

sum_XY = df['X*Y'].sum()
sum_XX = df['X^2'].sum()

# Calcular la pendiente (m) y el intercepto (b) utilizando la fórmula de regresión lineal
#num = 8 * np.sum(df['X'] * df['Y']) - np.sum(df['X']) * np.sum(df['Y'])
#den = 8 * np.sum(df['X'] ** 2) - (np.sum(df['X'])) ** 2
num = n * sum_XY - sum_X * sum_Y
den = n * sum_XX - (sum_X) ** 2

m = num / den
b = prom_Y - m * prom_X

resultado = f"El valor de m es igual:  {m} y el valor de b es iugal a: {b}. "
form_pendiente = f"Por lo tanto tendriamos y = {m}*x + {b}"

print(num)
print(den)
print(resultado)
print(form_pendiente)

#Gráfica
x_valores = np.linspace(min(df['X']), max(df['X']))
y_valores = m * x_valores + b
y = datos['Y']
x = datos['X']

plt.scatter(x,y)

plt.plot(x_valores, y_valores, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión Lineal')
plt.legend()
plt.show()






