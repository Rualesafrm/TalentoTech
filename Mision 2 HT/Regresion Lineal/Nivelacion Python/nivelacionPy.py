import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ruta = "D:/Nube/OneDrive - unicauca.edu.co/Cursos Online/Analisis de Datos/TalentoTech/Mision 2 HT/Regresion Lineal/Nivelacion Python/dataset_banco.csv"
data = pd.read_csv(ruta)

print(data.shape) #Me da el total de registros

print(data.head()) #Para ver las primeras filas del DataFrame 

print(data.info()) #Nombre de columnas, tipo de datos, total de registros por columna

#Datos fatantes em alguna celda. Celdas en blanco.
#Columnas irrelevantes.
#Registros repetidos. Eliminar filas repetidas
#Valores extremos. Edades de 200 años. outliers = valores extremos.
#Errores tipograficos. Married -> married

print(data.dropna(inplace = True)) #Elimina la fila completa donde encuentre datos faltantes
print(data.info()) #Nombre de columnas, tipo de datos, total de registros por columna

cols_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

for col in cols_cat:
    print(f'columna {col}: {data[col].nunique()} subniveles') #Un subnivel es una categoria de la columna por ejmplo si marital serian si es casado, soltero, viudo, etc.

#Si la desviacion estandar es cero quiere decir que no es importante, ya que me esta diciendo que todos esos datos de la columna son ceros
print(data.describe())#Nos deja ver todas las columnas numericas

print(f'Tamaño del set antes de eliminar las filas repetidas: {data.shape}')

data.drop_duplicates(inplace = True)#Elimina las filas duplicadas.

print(f'Tamaño del set despues de eliminar las filas repetidas: {data.shape}')

cols_num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

fig, ax = plt.subplots(nrows = 7, ncols = 1, figsize = (8, 30))#Crea una figura de 7 subgraficos dispuesto en una columna.

fig.subplots_adjust(hspace = 0.5)#Ajusta el espaciado vertical entre subgraficos

for i, col in enumerate(cols_num): #La función enumerate toma una lista (en este caso cols_num) y devuelve un objeto iterable que produce pares de (índice, valor) para cada elemento en la lista. El índice es un número entero que representa la posición del elemento en la lista, y el valor es el propio elemento.El bucle for itera sobre los elementos producidos por enumerate(cols_num). En cada iteración, i toma el valor del índice del elemento actual en la lista, y col toma el valor del elemento actual en la lista.
    sns.boxplot(x = col, data = data, ax = ax[i])
    ax[i].set_title(col)

plt.show() # Muestra las visualizaciones

#Filtrar o limpiar la edad donde se muestre solo las personas menores o iguales a 100 años
print(f'Tamaño del set antes de depurar la edad: {data.shape}')
data = data[data['age'] <= 100]
print(f'Tamaño del set despues de depurar o de filtrar por las personas menores o iguales a 100 años: {data.shape}')

#Filtrar o limpiar la duracion de las llamadas donde los minutos sean mayores a cero
print(f'Tamaño del set antes de depurar la duracion de los minutos: {data.shape}')
data = data[data['duration'] > 0]
print(f'Tamaño del set despues de depurar o de filtrar por las llamadas mayores a cero minutos: {data.shape}')


