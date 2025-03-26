# @title Proceso de análisis jerárquico Factores
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log, exp

# Función para calcular la media geométrica de una lista de números
def geometric_mean(values):
    # Evitar ceros; se asume que las intensidades son >=1
    logs = [log(v) for v in values if v > 0]
    return exp(np.mean(logs)) if logs else 0

# Ruta del CSV con las respuestas de los expertos
ruta_csv = "respuestas.csv"
df = pd.read_csv(ruta_csv)

# Definición de los 5 criterios a partir de las preguntas 1 a 10 (dimensiones generales)
criterios = ["Costos", "Barreras comerciales", "Logística", "Entorno y cultura", "Económico"]

# Mapeo de las preguntas a las posiciones de la matriz de comparación
# Se tienen 10 comparaciones que llenan la matriz de 5x5 (los índices corresponden al orden de 'criterios')
comparaciones_preg = {
    1: (0, 1),   # Q1: Costos vs Barreras comerciales
    2: (0, 2),   # Q2: Costos vs Logística
    3: (0, 3),   # Q3: Costos vs Entorno y cultura
    4: (0, 4),   # Q4: Costos vs Económico
    5: (1, 2),   # Q5: Barreras comerciales vs Logística
    6: (1, 3),   # Q6: Barreras comerciales vs Entorno y cultura
    7: (1, 4),   # Q7: Barreras comerciales vs Económico
    8: (2, 3),   # Q8: Logística vs Entorno y cultura
    9: (2, 4),   # Q9: Logística vs Económico
    10: (3, 4)   # Q10: Entorno y cultura vs Económico
}

# Para cada pregunta, vamos a agregar las respuestas de todos los expertos
# La idea es transformar cada respuesta en un número:
# Si en la columna A{i} la respuesta es "0" => se escogió la primera opción, la valoración es la intensidad (Intense{i})
# Si es "1" => se escogió la segunda opción y se toma el recíproco (1/Intense{i})
# Luego se calcula la media geométrica para obtener el valor agregado para cada comparación.

# Inicializamos un diccionario para guardar la valoración agregada por pregunta
valoraciones_agregadas = {}

for i in range(1, 11):  # Preguntas 1 a 10
    col_A = f"A{i}"
    col_Intense = f"Intense{i}"
    # Lista para almacenar el valor numérico de cada experto para la pregunta i
    valores = []
    for index, row in df.iterrows():
        # Convertir la respuesta a cadena para comparar
        respuesta = str(row[col_A]).strip()
        intensidad = float(row[col_Intense])
        if respuesta == "0":
            valor = intensidad
        elif respuesta == "1":
            valor = 1.0 / intensidad
        else:
            # En caso de datos no válidos se omite la respuesta
            continue
        valores.append(valor)
    if valores:
        valoraciones_agregadas[i] = geometric_mean(valores)
    else:
        valoraciones_agregadas[i] = 1.0  # valor neutro

# Ahora construimos la matriz de comparación de pares (5x5) para las dimensiones generales.
n = len(criterios)
matriz = np.ones((n, n))

# Llenamos la matriz usando las valoraciones agregadas para cada comparación
for pregunta, (i, j) in comparaciones_preg.items():
    if pregunta in valoraciones_agregadas:
        valor = valoraciones_agregadas[pregunta]
        matriz[i, j] = valor
        matriz[j, i] = 1.0 / valor

# Mostrar la matriz
print("Matriz de comparación de pares (agregada):")
print(pd.DataFrame(matriz, index=criterios, columns=criterios))

# Cálculo de la prioridad (pesos) usando el método del vector propio
eigenvalues, eigenvectors = np.linalg.eig(matriz)
# El mayor eigenvalor (lambda_max) y el correspondiente eigenvector real:
max_index = np.argmax(eigenvalues.real)
lambda_max = eigenvalues.real[max_index]
vector_prop = np.array(eigenvectors[:, max_index].real)
# Normalizamos el vector de prioridades
pesos = vector_prop / np.sum(vector_prop)

# Cálculo del Índice de Consistencia (CI) y Consistencia Relativa (CR)
CI = (lambda_max - n) / (n - 1)
# Valor aleatorio (RI) para n=5 (según la tabla de Saaty)
RI = 1.12
CR = CI / RI

print("\nResultados del Análisis Jerárquico:")
print("Pesos (prioridades):")
for crit, peso in zip(criterios, pesos):
    print(f"  {crit}: {peso:.4f}")
print(f"\nλ_max: {lambda_max:.4f}")
print(f"Índice de Consistencia (CI): {CI:.4f}")
print(f"Índice de Consistencia Relativa (CR): {CR:.4f} {'-> Matriz consistente' if CR < 0.1 else '-> Matriz inconsistente'}")

# ---------------------------
# Gráficas
# ---------------------------

# 1. Graficar la matriz de comparación como heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(matriz, annot=True, cmap="YlGnBu", fmt=".3f",
            xticklabels=criterios, yticklabels=criterios)
plt.title("Matriz de comparación de pares (Dimensiones generales)")
plt.show()

# 2. Gráfico de barras para los pesos obtenidos
plt.figure(figsize=(8, 6))
plt.bar(criterios, pesos, color='skyblue')
plt.xlabel("Criterios")
plt.ylabel("Peso")
plt.title("Pesos de los criterios (Prioridades)")
plt.ylim(0, max(pesos)*1.2)
for i, v in enumerate(pesos):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.show()

# 3. Opcional: mostrar la matriz normalizada (cada columna dividida por la suma de esa columna)
matriz_normalizada = matriz / matriz.sum(axis=0)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_normalizada, annot=True, cmap="coolwarm", fmt=".3f",
            xticklabels=criterios, yticklabels=criterios)
plt.title("Matriz normalizada de comparación de pares")
plt.show()

# Crear un DataFrame con los resultados del análisis jerárquico
resultados_df = pd.DataFrame({
    "Criterio": criterios,
    "Peso": pesos
})

# Guardar los resultados en un archivo CSV
ruta_resultados_csv = "pesos_factores.csv"
resultados_df.to_csv(ruta_resultados_csv, index=False)

print(f"\nLos resultados del Análisis Jerárquico se han guardado en: {ruta_resultados_csv}")

# Add heatmap image
plt.figure(figsize=(8, 6))
sns.heatmap(matriz, annot=True, cmap="YlGnBu", fmt=".3f",
            xticklabels=criterios, yticklabels=criterios)
plt.title("Matriz de comparación de pares (Dimensiones generales)")
plt.savefig("heatmap.png")
plt.close()

# Add bar chart image
plt.figure(figsize=(8, 6))
plt.bar(criterios, pesos, color='skyblue')
plt.xlabel("Criterios")
plt.ylabel("Peso")
plt.title("Pesos de los criterios (Prioridades)")
plt.ylim(0, max(pesos) * 1.2)
for i, v in enumerate(pesos):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.savefig("bar_chart.png")
plt.close()