# @title Modelo Topsis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar los archivos CSV
# Se asume que "pesos_subfactores.csv" contiene dos columnas: 'Subfactor' y 'Peso'
# y que "Topsis.csv" tiene en la primera columna el nombre del país y las siguientes columnas los subfactores.
pesos_df = pd.read_csv("pesos_subfactores.csv")
topsis_df = pd.read_csv("Topsis.csv")

# 2. Preparar la matriz de decisión
# Extraer la columna de nombres de país y los datos numéricos
paises = topsis_df.iloc[:, 0]
datos = topsis_df.iloc[:, 1:]

# 3. Extraer los pesos en el mismo orden que aparecen los subfactores en "datos"
# Se busca en pesos_df el peso correspondiente a cada subfactor
pesos = []
for col in datos.columns:
    # Buscar el peso del subfactor
    peso = pesos_df.loc[pesos_df['Subfactor'] == col, 'Peso']
    if not peso.empty:
        pesos.append(peso.values[0])
    else:
        print(f"Advertencia: No se encontró el peso para el subfactor '{col}'. Se asigna valor 1 por defecto.")
        pesos.append(1)  # Valor por defecto en caso de no encontrar el peso

pesos = np.array(pesos)

# 4. Normalización de la matriz de decisión
# Se utiliza la normalización Euclidiana para cada columna
norm_datos = datos / np.sqrt((datos**2).sum())

# 5. Construir la matriz ponderada
# Se multiplica cada columna normalizada por su peso respectivo
datos_ponderados = norm_datos * pesos

# 6. Determinar las soluciones ideales positiva y negativa
# Se asume que todos los criterios son de beneficio (a mayor valor, mejor)
ideal_positivo = datos_ponderados.max()
ideal_negativo = datos_ponderados.min()

# 7. Calcular las distancias euclidianas de cada alternativa a las soluciones ideales
distancia_positiva = np.sqrt(((datos_ponderados - ideal_positivo) ** 2).sum(axis=1))
distancia_negativa = np.sqrt(((datos_ponderados - ideal_negativo) ** 2).sum(axis=1))

# 8. Calcular el índice de cercanía (score TOPSIS)
score = distancia_negativa / (distancia_positiva + distancia_negativa)

# 9. Agregar el score al DataFrame original y ordenar los países
topsis_df['Score'] = score
topsis_df['Pais'] = paises  # Agregar columna de países en caso de necesitarla para visualizar el ranking
ranking = topsis_df.sort_values(by='Score', ascending=False)

# 10. Mostrar el ranking de países con sus respectivos scores
print("Ranking de países para exportar cacao (mayor score = mejor opción):")
print(ranking[['Pais', 'Score']])

# Gráfico del ranking de países
plt.figure(figsize=(12, 6))
plt.bar(ranking['Pais'], ranking['Score'])
plt.xlabel('País')
plt.ylabel('Score TOPSIS')
plt.title('Ranking de Países para Exportar Cacao')
plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para una mejor visualización
plt.tight_layout()
plt.show()
