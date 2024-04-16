import pandas as pd

datos = {'X': [1, 2, 2, 3, 4, 4, 5, 6],
         'Y': [2, 3, 4, 4, 4, 6, 5, 7]}

df = pd.DataFrame(datos)

prom_XY = (df['X'] * df['Y']).mean()
prom_X = df['X'].mean()
prom_Y = df['Y'].mean()
prom_X_cuadrado = (df['X'] ** 2).mean()

#Formula cov/var
m = (prom_XY - prom_X * prom_Y) / (prom_X_cuadrado - prom_X ** 2)

print("El valor de m es igual a:", m)