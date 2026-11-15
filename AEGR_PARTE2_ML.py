"""
PARTE 2 - Hiper-heurística de selección: entrenamiento y evaluación.

Uso:  python parte2_ml.py [carpeta_con_train_labeled_y_test_labeled] [carpeta_salida]

Modelos: Árbol de Decisión y KNN (hiperparámetros por validación cruzada
estratificada SOLO dentro de train; test se usa una única vez al final).
Métricas: accuracy vs. baseline trivial, F1 macro, matriz de confusión, regret
(absoluto y relativo) frente al oráculo y al mejor algoritmo único, accuracy por
rango de n, importancia de características y tabla (n, distribución) -> ganador.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
NUM = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
CAT = ["distribution"]
COST = {"2opt_hill_climbing": "cost_2opt_hill_climbing",
        "sa_2opt": "cost_sa_2opt", "mst_2opt": "cost_mst_2opt"}
CLASES = ["2opt_hill_climbing", "mst_2opt", "sa_2opt"]

base = sys.argv[1] if len(sys.argv) > 1 else "."
out = sys.argv[2] if len(sys.argv) > 2 else "resultados"
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "resultados.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


tr = pd.read_csv(os.path.join(base, "train_labeled.csv"))
te = pd.read_csv(os.path.join(base, "test_labeled.csv"))
todo = pd.concat([tr, te], ignore_index=True)

# ------------------------------------------------------------ balance de clases
P("=" * 70, "\n1. BALANCE DE CLASES (mis etiquetas)\n", "=" * 70)
bal = pd.DataFrame({"n": todo.best_algorithm.value_counts(),
                    "%": (100 * todo.best_algorithm.value_counts(normalize=True)).round(1)})
P(bal.to_string())
dominante = tr.best_algorithm.value_counts().idxmax()   # baseline con TRAIN (sin mirar test)
base_acc = (te.best_algorithm == dominante).mean()
P(f"\nBaseline trivial (siempre '{dominante}', elegido con train): accuracy test = {base_acc:.3f}")

empates = ((todo[list(COST.values())].max(axis=1) - todo[list(COST.values())].min(axis=1)) < 1e-9)
P(f"Instancias con los 3 costos iguales (empate total): {empates.sum()} "
  f"({100 * empates.mean():.1f}%), concentradas en n pequeño; "
  "se etiquetan como 2opt por la regla de desempate.")

# ------------------------------------------------------------ modelos
def preprocesador(escalar, usar_dist=True):
    cat = [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)] if usar_dist else []
    num = [("num", StandardScaler() if escalar else "passthrough", NUM)]
    return ColumnTransformer(num + cat)


def construir(nombre, usar_dist=True, balanced=False):
    cw = "balanced" if balanced else None
    if nombre == "Arbol":
        pipe = Pipeline([("pre", preprocesador(False, usar_dist)),
                         ("clf", DecisionTreeClassifier(random_state=SEED, class_weight=cw))])
        grid = {"clf__max_depth": [2, 3, 4, 5, 6, 8],
                "clf__min_samples_leaf": [1, 3, 5, 10, 20]}
    else:
        pipe = Pipeline([("pre", preprocesador(True, usar_dist)),
                         ("clf", KNeighborsClassifier())])
        grid = {"clf__n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31],
                "clf__weights": ["uniform", "distance"]}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro" if balanced else "accuracy",
                        n_jobs=1)


def regret(df, pred):
    costos = df[list(COST.values())].to_numpy()
    elegido = np.array([df.iloc[i][COST[p]] for i, p in enumerate(pred)])
    mejor = costos.min(axis=1)
    return elegido - mejor, (elegido - mejor) / mejor


def evaluar(nombre, modelo, X_te, y_te, df_te):
    pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro", labels=CLASES)
    rec = recall_score(y_te, pred, average=None, labels=CLASES, zero_division=0)
    ra, rr = regret(df_te, pred)
    return dict(modelo=nombre, accuracy=acc, f1_macro=f1,
                recall_2opt=rec[0], recall_mst=rec[1], recall_sa=rec[2],
                regret_medio=ra.mean(), regret_mediana=np.median(ra), regret_max=ra.max(),
                regret_rel_medio_pct=100 * rr.mean()), pred


X_tr, y_tr, X_te, y_te = tr[NUM + CAT], tr.best_algorithm, te[NUM + CAT], te.best_algorithm

configs = [("Arbol", True, False), ("Arbol (class_weight=balanced)", True, True),
           ("Arbol sin 'distribution'", False, False),
           ("KNN", True, False), ("KNN sin 'distribution'", False, False)]
filas, preds, modelos = [], {}, {}
P("\n", "=" * 70, "\n2. ENTRENAMIENTO (CV 5-fold estratificada dentro de train)\n", "=" * 70)
for etiqueta, usar_dist, bal_cw in configs:
    tipo = "Arbol" if etiqueta.startswith("Arbol") else "KNN"
    gs = construir(tipo, usar_dist, bal_cw)
    gs.fit(X_tr, y_tr)
    P(f"{etiqueta:34s} mejores params: {gs.best_params_}  CV={gs.best_score_:.3f}")
    r, p = evaluar(etiqueta, gs.best_estimator_, X_te, y_te, te)
    filas.append(r)
    preds[etiqueta] = p
    modelos[etiqueta] = gs.best_estimator_

# referencias: baseline y oráculo
ra, rr = regret(te, [dominante] * len(te))
filas.insert(0, dict(modelo=f"BASELINE siempre {dominante}", accuracy=base_acc,
                     f1_macro=f1_score(y_te, [dominante] * len(te), average="macro", labels=CLASES),
                     recall_2opt=1.0, recall_mst=0.0, recall_sa=0.0, regret_medio=ra.mean(),
                     regret_mediana=np.median(ra), regret_max=ra.max(),
                     regret_rel_medio_pct=100 * rr.mean()))
filas.insert(1, dict(modelo="ORACULO", accuracy=1.0, f1_macro=1.0, recall_2opt=1.0,
                     recall_mst=1.0, recall_sa=1.0, regret_medio=0.0, regret_mediana=0.0,
                     regret_max=0.0, regret_rel_medio_pct=0.0))
tabla = pd.DataFrame(filas).set_index("modelo")
tabla.round(4).to_csv(os.path.join(out, "tabla_comparativa_test.csv"))

P("\n", "=" * 70, "\n3. RESULTADOS EN TEST (149 instancias)\n", "=" * 70)
P(tabla.round(4).to_string())

# ------------------------------------------------------------ modelo principal
principal = max([f["modelo"] for f in filas[2:] if "sin" not in f["modelo"]],
                key=lambda m: tabla.loc[m, "accuracy"])
P(f"\nModelo principal para el análisis (mayor accuracy con 'distribution'): {principal}")
pred = preds[principal]
cm = confusion_matrix(y_te, pred, labels=CLASES)
P("\nMatriz de confusión (filas = real, columnas = predicho)")
P(pd.DataFrame(cm, index=CLASES, columns=CLASES).to_string())

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(CLASES, rotation=20)
ax.set_yticks(range(3)); ax.set_yticklabels(CLASES)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión - {principal}")
plt.tight_layout(); plt.savefig(os.path.join(out, "matriz_confusion.png"), dpi=150); plt.close()

# regret detallado
ra, rr = regret(te, pred)
ra_b, _ = regret(te, [dominante] * len(te))
P("\nREGRET (costo elegido - costo del mejor real), escala de coordenadas [0,1]")
P(f"  Oráculo:                 0")
P(f"  Baseline (siempre {dominante}): medio={ra_b.mean():.5f}")
P(f"  Modelo ({principal}): medio={ra.mean():.5f}  mediana={np.median(ra):.5f}  "
  f"max={ra.max():.5f}  relativo medio={100 * rr.mean():.3f}%")
P(f"  Reducción del regret medio vs. baseline: {100 * (1 - ra.mean() / ra_b.mean()):.1f}%")

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([ra_b, ra], tick_labels=["Baseline", principal], showfliers=True)
ax.set_ylabel("Regret absoluto (test)"); ax.set_title("Regret en test")
ax.grid(axis="y", ls="--", alpha=.6); plt.tight_layout()
plt.savefig(os.path.join(out, "regret.png"), dpi=150); plt.close()

# accuracy / regret por rango de n
rango = pd.cut(te.n, [0, 10, 30, 60, 100], labels=["n<=10", "15-30", "40-60", "80-100"])
por_n = pd.DataFrame({"rango_n": rango, "acierto": (pred == y_te.values),
                      "acierto_baseline": (y_te.values == dominante),
                      "regret": ra, "regret_baseline": ra_b}).groupby("rango_n").agg(
    instancias=("acierto", "size"), acc_modelo=("acierto", "mean"),
    acc_baseline=("acierto_baseline", "mean"), regret_modelo=("regret", "mean"),
    regret_baseline=("regret_baseline", "mean"))
P("\nDESEMPEÑO POR RANGO DE n (test)")
P(por_n.round(4).to_string())

# ------------------------------------------------------------ importancia
P("\n", "=" * 70, "\n4. IMPORTANCIA DE CARACTERÍSTICAS\n", "=" * 70)
m = modelos[principal]
if hasattr(m.named_steps["clf"], "feature_importances_"):
    nombres = m.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(m.named_steps["clf"].feature_importances_, index=nombres).sort_values(ascending=False)
    P("Importancia del árbol (impureza):\n" + imp.round(4).to_string())
pi = permutation_importance(m, X_te, y_te, n_repeats=30, random_state=SEED, scoring="accuracy")
pi_s = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values(ascending=False)
P("\nImportancia por permutación (caída de accuracy en test):\n" + pi_s.round(4).to_string())
bal_name = "Arbol (class_weight=balanced)"
mb = modelos[bal_name]
P(f"\n[Nota] El modelo principal puede colapsar a una sola clase, y entonces su importancia por "
  f"permutación es 0 por construcción. Se repite el análisis con '{bal_name}', que sí predice las 3 clases:")
nombres_b = mb.named_steps["pre"].get_feature_names_out()
imp_b = pd.Series(mb.named_steps["clf"].feature_importances_, index=nombres_b).sort_values(ascending=False)
P("Importancia del árbol balanceado (impureza):\n" + imp_b.round(4).to_string())
pib = permutation_importance(mb, X_te, y_te, n_repeats=30, random_state=SEED, scoring="f1_macro")
pib_s = pd.Series(pib.importances_mean, index=NUM + CAT).sort_values(ascending=False)
P("Importancia por permutación del árbol balanceado (caída de F1 macro en test):\n" + pib_s.round(4).to_string())
fig, ax = plt.subplots(figsize=(6, 3.5))
pib_s[::-1].plot.barh(ax=ax, color="#55a868")
ax.set_xlabel("Caída de F1 macro al permutar"); ax.set_title("Importancia - Árbol balanceado")
plt.tight_layout(); plt.savefig(os.path.join(out, "importancia_balanceado.png"), dpi=150); plt.close()
P("\nCorrelación entre características numéricas (train):")
P(tr[NUM].corr().round(2).to_string())

fig, ax = plt.subplots(figsize=(6, 3.5))
pi_s[::-1].plot.barh(ax=ax, color="#3d85c0")
ax.set_xlabel("Caída de accuracy al permutar"); ax.set_title("Importancia por permutación")
plt.tight_layout(); plt.savefig(os.path.join(out, "importancia.png"), dpi=150); plt.close()

# ------------------------------------------------------------ patrones
P("\n", "=" * 70, "\n5. PATRÓN (n, distribución) -> ALGORITMO GANADOR (500 instancias)\n", "=" * 70)
P("Por distribución:")
P(pd.crosstab(todo.distribution, todo.best_algorithm, normalize="index").mul(100).round(1).to_string())
P("\nPor tamaño n:")
P(pd.crosstab(todo.n, todo.best_algorithm, normalize="index").mul(100).round(1).to_string())
P("\nPor tamaño n y distribución (ganador más frecuente):")
tabla_g = todo.groupby(["n", "distribution"]).best_algorithm.agg(
    lambda s: f"{s.value_counts().idxmax()} ({100 * s.value_counts(normalize=True).max():.0f}%)").unstack()
P(tabla_g.to_string())

# ¿mejora SA / MST sobre 2-opt cuando gana? magnitud
g = todo.copy()
g["mejora_pct"] = 100 * (g["cost_2opt_hill_climbing"] - g[list(COST.values())].min(axis=1)) / g["cost_2opt_hill_climbing"]
P("\nMejora relativa del mejor algoritmo sobre 2-opt, por ganador (%):")
P(g.groupby("best_algorithm").mejora_pct.describe().round(2).to_string())

tiempos = [c for c in todo.columns if c.startswith("time_")]
P("\nTiempo medio de ejecución (s) por algoritmo:")
P(todo[tiempos].mean().round(5).to_string())
log.close()
print(f"\nResultados guardados en: {out}/")
