# @title Proceso de análisis jerárquico de los Subfactores
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log, exp

# Función para calcular la media geométrica de una lista de números
def geometric_mean(values):
    logs = [log(v) for v in values if v > 0]
    return exp(np.mean(logs)) if logs else 0

# Función que procesa un grupo de subfactores dado el rango de preguntas,
# el diccionario de comparación (mapeo de pregunta a par de subfactores) y el nombre del grupo.
def process_group(df, preguntas, mapping, group_name):
    # Agregamos las valoraciones de cada pregunta mediante la media geométrica
    agregados = {}
    for i in preguntas:
        if i in mapping:
            col_A = f"A{i}"
            col_Intense = f"Intense{i}"
            valores = []
            for index, row in df.iterrows():
                respuesta = str(row[col_A]).strip()
                intensidad = float(row[col_Intense])
                if respuesta == "0":
                    valor = intensidad
                elif respuesta == "1":
                    valor = 1.0 / intensidad
                else:
                    continue
                valores.append(valor)
            if valores:
                agregados[i] = geometric_mean(valores)
            else:
                agregados[i] = 1.0  # valor neutro
    # Extraer el conjunto de subfactores a partir de los pares en las preguntas del grupo
    subcriterios = set()
    for i in preguntas:
        if i in mapping:
            subcriterios.update(mapping[i])
    subcriterios = sorted(list(subcriterios))
    n = len(subcriterios)

    # Inicializar la matriz de comparación con unos (valor neutro en AHP)
    matriz = np.ones((n, n))
    # Mapear cada subfactores a un índice
    idx_map = {sc: idx for idx, sc in enumerate(subcriterios)}

    # Llenar la matriz con los valores agregados para cada comparación disponible
    for i in preguntas:
        if i in agregados and i in mapping:
            sc1, sc2 = mapping[i]
            if sc1 in idx_map and sc2 in idx_map:
                valor = agregados[i]
                matriz[idx_map[sc1], idx_map[sc2]] = valor
                matriz[idx_map[sc2], idx_map[sc1]] = 1.0 / valor

    # Cálculo de prioridades (pesos) usando el método del vector propio
    eigenvalues, eigenvectors = np.linalg.eig(matriz)
    max_index = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues.real[max_index]
    vector_prop = eigenvectors[:, max_index].real
    pesos = vector_prop / np.sum(vector_prop)

    # Cálculo del Índice de Consistencia (CI) y Consistencia Relativa (CR)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_dict.get(n, 1.45)
    CR = CI / RI if RI != 0 else 0

    # Mostrar resultados en consola
    print(f"Resultados para {group_name}:")
    print("Subfactores:", subcriterios)
    for sc, peso in zip(subcriterios, pesos):
        print(f"  {sc}: {peso:.4f}")
    print(f"λ_max: {lambda_max:.4f}")
    print(f"CI: {CI:.4f}, CR: {CR:.4f} {'-> Matriz consistente' if CR < 0.1 else '-> Matriz inconsistente'}\n")

    # Gráfica 1: Heatmap de la matriz de comparación
    plt.figure(figsize=(8,6))
    sns.heatmap(matriz, annot=True, cmap="YlGnBu", fmt=".3f",
                xticklabels=subcriterios, yticklabels=subcriterios)
    plt.title(f"Matriz de comparación de pares para {group_name}")
    plt.show()

    # Gráfica 2: Gráfico de barras de los pesos
    plt.figure(figsize=(8,6))
    plt.bar(subcriterios, pesos, color="skyblue")
    plt.xlabel("Subfactores")
    plt.ylabel("Peso")
    plt.title(f"Pesos de los subfactores para {group_name}")
    plt.ylim(0, max(pesos)*1.2)
    for i, v in enumerate(pesos):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.show()

    # Devolver los resultados de consistencia
    return {"Grupo": group_name, "CI": CI, "CR": CR}

# Cargar el CSV con las respuestas de los expertos
ruta_csv = "respuestas.csv"
df = pd.read_csv(ruta_csv)

# Diccionario de mapeo para las comparaciones de subfactores (según el formulario)
mapping_sub = {
    # Costos (preguntas 11 a 20)
    11: ("Precio en destino", "Precio en transporte internacional"),
    12: ("Precio en destino", "Costo de importación"),
    13: ("Precio en destino", "Incidencia del tipo de cambio"),
    14: ("Precio en destino", "Transporte interno"),
    15: ("Precio en transporte internacional", "Costo de importación"),
    16: ("Precio en transporte internacional", "Incidencia del tipo de cambio"),
    17: ("Precio en transporte internacional", "Transporte interno"),
    18: ("Costo de importación", "Incidencia del tipo de cambio"),
    19: ("Costo de importación", "Transporte interno"),
    20: ("Incidencia del tipo de cambio", "Transporte interno"),
    # Barreras comerciales (preguntas 21 a 30)
    21: ("Aranceles", "Proteccionismo en general"),
    22: ("Aranceles", "Índice de libertad económica"),
    23: ("Aranceles", "Barreras no arancelarias"),
    24: ("Aranceles", "Competencia internacional"),
    25: ("Proteccionismo en general", "Índice de libertad económica"),
    26: ("Proteccionismo en general", "Barreras no arancelarias"),
    27: ("Proteccionismo en general", "Competencia internacional"),
    28: ("Índice de libertad económica", "Barreras no arancelarias"),
    29: ("Índice de libertad económica", "Competencia internacional"),
    30: ("Barreras no arancelarias", "Competencia internacional"),
    # Logística (preguntas 31 a 40)
    31: ("Índice de desempeño logístico", "Tiempo de transito"),
    32: ("Índice de desempeño logístico", "Frecuencia"),
    33: ("Índice de desempeño logístico", "Distancia geográfica"),
    34: ("Índice de desempeño logístico", "Ubicación geográfica"),
    35: ("Tiempo de transito", "Frecuencia"),
    36: ("Tiempo de transito", "Distancia geográfica"),
    37: ("Tiempo de transito", "Ubicación geográfica"),
    38: ("Frecuencia", "Distancia geográfica"),
    39: ("Frecuencia", "Ubicación geográfica"),
    40: ("Distancia geográfica", "Ubicación geográfica"),
    # Entorno y cultura (preguntas 41 a 46)
    41: ("Facilidad para hacer negocios", "Índice de percepción de corrupción"),
    42: ("Facilidad para hacer negocios", "Des-afinidad cultural"),
    43: ("Facilidad para hacer negocios", "Índice de globalización"),
    44: ("Índice de percepción de corrupción", "Des-afinidad cultural"),
    45: ("Índice de percepción de corrupción", "Índice de globalización"),
    46: ("Des-afinidad cultural", "Índice de globalización"),
    # Económico (preguntas 47 a 52)
    47: ("PIB per cápita", "Tasa de desempleo"),
    48: ("PIB per cápita", "Índice de costo de vida"),
    49: ("PIB per cápita", "Riesgo país"),
    50: ("Tasa de desempleo", "Índice de costo de vida"),
    51: ("Tasa de desempleo", "Riesgo país"),
    52: ("Índice de costo de vida", "Riesgo país")

}

# Procesar cada grupo de subfactores y almacenar los resultados de consistencia
print("=== PROCESO AHP PARA SUBFACTORES ===\n")
resultados_consistencia = []

resultados_consistencia.append(process_group(df, range(11, 21), mapping_sub, "Costos"))
resultados_consistencia.append(process_group(df, range(21, 31), mapping_sub, "Barreras comerciales"))
resultados_consistencia.append(process_group(df, range(31, 41), mapping_sub, "Logística"))
resultados_consistencia.append(process_group(df, range(41, 47), mapping_sub, "Entorno y cultura"))
resultados_consistencia.append(process_group(df, range(47, 53), mapping_sub, "Económico"))

# Mostrar un resumen de los índices de consistencia
print("\n=== RESUMEN DE ÍNDICES DE CONSISTENCIA ===")
for resultado in resultados_consistencia:
    print(f"Grupo: {resultado['Grupo']}, CI: {resultado['CI']:.4f}, CR: {resultado['CR']:.4f} {'(Consistente)' if resultado['CR'] < 0.1 else '(Inconsistente)'}")

# Guardar los pesos en un archivo CSV
output_csv = "pesos_subfactores.csv"
with open(output_csv, 'w') as f:
    f.write("Grupo,Subfactor,Peso\n") # Escribir encabezado

def process_group_for_csv(df, preguntas, mapping, group_name, output_file):
    agregados = {}
    for i in preguntas:
        if i in mapping:
            col_A = f"A{i}"
            col_Intense = f"Intense{i}"
            valores = []
            for index, row in df.iterrows():
                respuesta = str(row[col_A]).strip()
                intensidad = float(row[col_Intense])
                if respuesta == "0":
                    valor = intensidad
                elif respuesta == "1":
                    valor = 1.0 / intensidad
                else:
                    continue
                valores.append(valor)
            if valores:
                agregados[i] = geometric_mean(valores)
            else:
                agregados[i] = 1.0

    subcriterios = sorted(set(sc for i in preguntas if i in mapping for sc in mapping[i]))
    n = len(subcriterios)
    matriz = np.ones((n, n))
    idx_map = {sc: idx for idx, sc in enumerate(subcriterios)}

    for i in preguntas:
        if i in agregados and i in mapping:
            sc1, sc2 = mapping[i]
            if sc1 in idx_map and sc2 in idx_map:
                valor = agregados[i]
                matriz[idx_map[sc1], idx_map[sc2]] = valor
                matriz[idx_map[sc2], idx_map[sc1]] = 1.0 / valor

    eigenvalues, eigenvectors = np.linalg.eig(matriz)
    max_index = np.argmax(eigenvalues.real)
    vector_prop = eigenvectors[:, max_index].real
    pesos = vector_prop / np.sum(vector_prop)

    for sc, peso in zip(subcriterios, pesos):
        with open(output_file, 'a') as f:
            f.write(f"{group_name},{sc},{peso:.4f}\n")

process_group_for_csv(df, range(11, 21), mapping_sub, "Costos", output_csv)
process_group_for_csv(df, range(21, 31), mapping_sub, "Barreras comerciales", output_csv)
process_group_for_csv(df, range(31, 41), mapping_sub, "Logística", output_csv)
process_group_for_csv(df, range(41, 47), mapping_sub, "Entorno y cultura", output_csv)
process_group_for_csv(df, range(47, 53), mapping_sub, "Económico", output_csv)