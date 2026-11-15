"""
PARTE 1 - Generación de etiquetas (hiper-heurística de selección para TSP).

Corre los tres algoritmos de la Tarea 2 sobre cada instancia de train.csv y
test.csv y agrega: cost_2opt_hill_climbing, cost_sa_2opt, cost_mst_2opt,
best_algorithm (más tiempos de ejecución como información extra).

Uso:  python parte1_etiquetas.py [ruta_a_hh_tsp_dataset/data] [carpeta_salida]

DECISIONES DE ETIQUETADO (documentar en el reporte)
---------------------------------------------------
* Escala: las instancias están en [0,1]^2, pero los parámetros de SA de la
  Tarea 2 (T0=1000, T_min=1e-3, alpha=0.9995) fueron calibrados para
  coordenadas en [0,100]. Para conservarlos tal cual, las coordenadas se
  multiplican por 100 al correr los algoritmos y los costos se reportan de
  vuelta en la escala original [0,1] (costo / 100).
* 2-opt hill climbing: solución inicial = vecino más cercano desde el nodo 0
  (determinista); luego 2-opt hasta no encontrar mejoras.
* SA + 2-opt: enfriamiento exponencial (mejor esquema en la Tarea 2), T0=1000,
  alpha=0.9995, T_min=1e-3 (~27 600 iteraciones), tour inicial aleatorio,
  movimiento 2-opt estocástico, criterio de Metropolis. Se devuelve el MEJOR
  tour visitado y su costo se recalcula desde cero (sin deriva numérica).
* MST + 2-opt: MST por Prim, recorrido DFS en preorden (2-aproximación) y
  después 2-opt. Prim propio (no scipy) para tolerar puntos duplicados.
* Semilla: una por instancia, derivada del número de instance_id, para que el
  resultado no dependa del orden de ejecución.
* Desempate: si la diferencia relativa entre costos es < 1e-9 se consideran
  empatados y gana el de menor prioridad de costo computacional:
  2opt_hill_climbing > mst_2opt > sa_2opt.
"""
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd

ESCALA = 100.0
T0, T_MIN, ALPHA = 1000.0, 1e-3, 0.9995     # parámetros SA (Tarea 2)
TOL_EMPATE = 1e-9
PRIORIDAD = ["cost_2opt_hill_climbing", "cost_mst_2opt", "cost_sa_2opt"]
NOMBRE = {
    "cost_2opt_hill_climbing": "2opt_hill_climbing",
    "cost_mst_2opt": "mst_2opt",
    "cost_sa_2opt": "sa_2opt",
}


# ---------------------------------------------------------------- utilidades
def matriz_dist(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def costo_tour(tour, d):
    return float(sum(d[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour))))


def es_permutacion(tour, n):
    return len(tour) == n and sorted(tour) == list(range(n))


# ---------------------------------------------------------------- algoritmos
def vecino_mas_cercano(d, inicio=0):
    n = len(d)
    visitados = [False] * n
    visitados[inicio] = True
    tour, actual = [inicio], inicio
    for _ in range(n - 1):
        mejor, mejor_d = -1, math.inf
        fila = d[actual]
        for j in range(n):
            if not visitados[j] and fila[j] < mejor_d:
                mejor, mejor_d = j, fila[j]
        visitados[mejor] = True
        tour.append(mejor)
        actual = mejor
    return tour


def dos_opt(tour, d):
    """Búsqueda local 2-opt (hill climbing) hasta óptimo local."""
    n = len(tour)
    tour = list(tour)
    if n < 4:
        return tour
    mejoro = True
    while mejoro:
        mejoro = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, e = tour[j], tour[(j + 1) % n]
                delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
                if delta < -1e-9:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    mejoro = True
    return tour


def algoritmo_2opt(d, rng):
    return dos_opt(vecino_mas_cercano(d), d)


def algoritmo_sa(d, rng):
    n = len(d)
    if n < 4:
        return list(range(n))
    actual = list(range(n))
    rng.shuffle(actual)
    c_actual = costo_tour(actual, d)
    mejor, c_mejor = list(actual), c_actual
    T = T0
    while T > T_MIN:
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        if not (i == 0 and j == n - 1):
            a, b = actual[i], actual[i + 1]
            c, e = actual[j], actual[(j + 1) % n]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            if delta < 0 or rng.random() < math.exp(-delta / T):
                actual[i + 1:j + 1] = actual[i + 1:j + 1][::-1]
                c_actual += delta
                if c_actual < c_mejor - 1e-12:
                    c_mejor, mejor = c_actual, list(actual)
        T *= ALPHA
    return mejor


def algoritmo_mst_2opt(d, rng):
    n = len(d)
    # Prim
    en_arbol = [False] * n
    mejor_arista = [math.inf] * n
    padre = [-1] * n
    mejor_arista[0] = 0.0
    adj = [[] for _ in range(n)]
    for _ in range(n):
        u = min((v for v in range(n) if not en_arbol[v]), key=lambda v: mejor_arista[v])
        en_arbol[u] = True
        if padre[u] != -1:
            adj[u].append(padre[u])
            adj[padre[u]].append(u)
        for v in range(n):
            if not en_arbol[v] and d[u][v] < mejor_arista[v]:
                mejor_arista[v], padre[v] = d[u][v], u
    # DFS preorden (iterativo)
    tour, visto, pila = [], [False] * n, [0]
    while pila:
        u = pila.pop()
        if visto[u]:
            continue
        visto[u] = True
        tour.append(u)
        for v in reversed(adj[u]):
            if not visto[v]:
                pila.append(v)
    return dos_opt(tour, d)


ALGORITMOS = {
    "cost_2opt_hill_climbing": algoritmo_2opt,
    "cost_sa_2opt": algoritmo_sa,
    "cost_mst_2opt": algoritmo_mst_2opt,
}


# ---------------------------------------------------------------- etiquetado
def mejor_algoritmo(costos):
    minimo = min(costos.values())
    for col in PRIORIDAD:  # primero en prioridad entre los empatados
        if costos[col] <= minimo * (1 + TOL_EMPATE) + 1e-15:
            return NOMBRE[col]


def etiquetar_instancia(ruta_csv, instance_id):
    coords = pd.read_csv(ruta_csv).to_numpy(dtype=float) * ESCALA
    n = len(coords)
    d = matriz_dist(coords).tolist()
    semilla = int(instance_id.split("_")[1])
    fila = {}
    for col, fn in ALGORITMOS.items():
        rng = random.Random(semilla * 1000 + list(ALGORITMOS).index(col))
        t0 = time.perf_counter()
        tour = fn(d, rng)
        fila["time_" + col[5:]] = time.perf_counter() - t0
        assert es_permutacion(tour, n), f"{instance_id}: {col} no devolvió una permutación válida"
        fila[col] = costo_tour(tour, d) / ESCALA   # de vuelta a escala [0,1]
    fila["best_algorithm"] = mejor_algoritmo({c: fila[c] for c in ALGORITMOS})
    return fila


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "hh_tsp_dataset/data"
    salida = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(salida, exist_ok=True)

    for nombre in ["train", "test"]:
        df = pd.read_csv(os.path.join(base, f"{nombre}.csv"))
        filas = []
        t0 = time.time()
        for k, iid in enumerate(df["instance_id"], 1):
            filas.append(etiquetar_instancia(os.path.join(base, "instances", f"{iid}.csv"), iid))
            if k % 50 == 0:
                print(f"[{nombre}] {k}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
        etiquetado = pd.concat([df, pd.DataFrame(filas)], axis=1)
        etiquetado.to_csv(os.path.join(salida, f"{nombre}_labeled.csv"), index=False)
        print(f"{nombre}_labeled.csv guardado ({len(etiquetado)} filas)")

    tr = pd.read_csv(os.path.join(salida, "train_labeled.csv"))
    te = pd.read_csv(os.path.join(salida, "test_labeled.csv"))
    total = pd.concat([tr, te])
    print("\nBALANCE DE CLASES (500 instancias)")
    print(pd.DataFrame({"n": total["best_algorithm"].value_counts(),
                        "%": (100 * total["best_algorithm"].value_counts(normalize=True)).round(1)}))


if __name__ == "__main__":
    main()
