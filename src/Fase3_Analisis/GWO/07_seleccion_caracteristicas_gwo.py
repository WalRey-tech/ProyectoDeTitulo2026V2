# -*- coding: utf-8 -*-
"""
Selección de Características con Metaheurística — GWO (Grey Wolf Optimizer)
============================================================================
Problema : seleccionar el subconjunto óptimo de features TF-IDF (400 dims)
           que maximice F1-macro en clasificación de perfiles de egreso.

Codificación : binaria — cada lobo = vector {0,1}^400
               1 = feature incluida, 0 = excluida

Fitness      : F1-macro (StratifiedKFold-3, SMOTE k=2 dentro del fold)
               Penalización si selecciona 0 features o >95% del espacio

Optimizador  : GWO.OriginalGWO (mealpy) — Mirjalili et al. (2014)

Referencias
-----------
  Mirjalili et al. (2014) — Grey Wolf Optimizer. Advances in Engineering Software.
  Emary et al. (2016)     — Binary grey wolf optimization approaches for feature
                             selection. Neurocomputing.
  Mafarja & Mirjalili (2017) — Hybrid Whale Optimization Algorithm with simulated
                             annealing for feature selection. Neurocomputing.
  Nguyen et al. (2020)    — mealpy: A Framework of Metaheuristic Algorithms in Python.
"""

import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ftfy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# Carpeta donde se encuentra este script:
# src/Fase3_Analisis/GWO/
BASE = os.path.dirname(os.path.abspath(__file__))

# Retrocedemos dos niveles para llegar a:
# src/
SRC_ROOT = os.path.abspath(
    os.path.join(BASE, "..", "..")
)

# Dataset V2 utilizado por la metodología GWO
CSV_V2 = os.path.join(
    SRC_ROOT,
    "data",
    "processed",
    "perfiles_egreso_etiquetado_v2.csv"
)

# Resultados generados por NUESTRA ejecución integrada
OUT_DIR = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "gwo"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Carga ──────────────────────────────────────────────────────────────────────
print("=" * 68)
print("SELECCION DE FEATURES CON GWO (Grey Wolf Optimizer)")
print("Mirjalili et al. (2014) + Binary encoding (Emary et al., 2016)")
print("=" * 68)

df = pd.read_csv(CSV_V2, encoding='utf-8-sig')
df['perfil_egreso'] = df['perfil_egreso'].apply(lambda x: ftfy.fix_text(str(x)))
df['grado']         = df['grado'].apply(lambda x: ftfy.fix_text(str(x)))

textos = df['perfil_egreso'].tolist()
le     = LabelEncoder()
y      = le.fit_transform(df['grado'].tolist())
print(f"Dataset: {len(df)} docs — {dict(df['grado'].value_counts())}")

# ── Vectorización TF-IDF (config idéntica al mejor modelo) ────────────────────
# Nota: TF-IDF se ajusta sobre todo el corpus (preprocesamiento).
# La selección con GWO y evaluación son independientes del ajuste del vectorizador.
# Stopwords en español — lista estándar NLTK + términos de dominio genérico
STOPWORDS_ES = [
    'a','al','algo','algunas','algunos','ante','antes','como','con','contra',
    'cual','cuando','de','del','desde','donde','durante','e','el','ella',
    'ellas','ellos','en','entre','era','erais','eran','eras','eres','es',
    'esa','esas','ese','eso','esos','esta','estaba','estaban','estado',
    'estar','estas','este','esto','estos','estoy','fue','fueron','fui',
    'ha','han','has','hasta','hay','he','hun','la','las','le','les','lo',
    'los','mas','me','mi','mia','mias','mientras','mis','mo','mucho',
    'muchos','muy','más','mí','nada','ni','no','nos','nosotras','nosotros',
    'nuestra','nuestras','nuestro','nuestros','o','os','otra','otras','otro',
    'otros','para','pero','poco','por','porque','que','quien','quienes',
    'qué','se','sea','seais','sean','seas','ser','si','sin','sobre','sois',
    'somos','son','su','sus','también','tanto','te','tenemos','tengo','ti',
    'tiene','tienen','toda','todas','todo','todos','tu','tus','un','una',
    'unas','uno','unos','vosotras','vosotros','vuestra','vuestras','vuestro',
    'vuestros','y','ya','yo','él','ésta','éstas','éste','éstos','última',
    'últimas','último','últimos',
]

VEC = TfidfVectorizer(max_features=400, ngram_range=(1, 2),
                      min_df=2, max_df=0.9, sublinear_tf=True,
                      stop_words=STOPWORDS_ES)
X_full = VEC.fit_transform(textos).toarray()   # (61, 400)
feature_names = np.array(VEC.get_feature_names_out())
N_FEATURES = X_full.shape[1]
print(f"Espacio de features: {N_FEATURES} dimensiones (TF-IDF word 1-2 grams)")

# ── CV interno para la función fitness ────────────────────────────────────────
# 3-fold (en vez de 5) para reducir tiempo por evaluación
INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

def fitness_fn_array(solution_bin):
    """
    Evalúa un subconjunto de features dado el vector binario.
    Retorna F1-macro (valor a MAXIMIZAR).
    """
    mask = solution_bin.astype(bool)
    n_sel = mask.sum()
    if n_sel == 0:
        return 0.0                # penalización: sin features
    if n_sel == N_FEATURES:
        return 0.0                # penalización: selecciona todo (sin reducción)

    X_sel = X_full[:, mask]
    f1s = []
    for tr, te in INNER_CV.split(X_sel, y):
        X_tr, X_te = X_sel[tr], X_sel[te]
        y_tr, y_te = y[tr], y[te]
        try:
            smote = SMOTE(k_neighbors=2, random_state=SEED)
            X_r, y_r = smote.fit_resample(X_tr, y_tr)
            clf = ComplementNB()
            clf.fit(X_r, y_r)
            f1s.append(f1_score(y_te, clf.predict(X_te),
                                average='macro', zero_division=0))
        except Exception:
            f1s.append(0.0)
    return float(np.mean(f1s))

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE (sin selección) — evaluado con mismo CV interno
# ══════════════════════════════════════════════════════════════════════════════
def evaluar_baseline_inner():
    """Evalúa las 400 variables sin activar la penalización de la función GWO."""
    f1s = []
    for tr, te in INNER_CV.split(X_full, y):
        X_tr, X_te = X_full[tr], X_full[te]
        y_tr, y_te = y[tr], y[te]
        smote = SMOTE(k_neighbors=2, random_state=SEED)
        X_r, y_r = smote.fit_resample(X_tr, y_tr)
        clf = ComplementNB()
        clf.fit(X_r, y_r)
        f1s.append(
            f1_score(y_te, clf.predict(X_te), average='macro', zero_division=0)
        )
    return float(np.mean(f1s))


f1_base = evaluar_baseline_inner()
print(f"\nBaseline (400 features, 3-fold inner CV): F1={f1_base:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# GWO — Grey Wolf Optimizer (versión binaria con función de transferencia V-shape)
# Emary et al. (2016): posiciones continuas → binarias vía T(x) = |2/π·arctan(π/2·x)|
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 68)
print("GWO — Grey Wolf Optimizer (Mirjalili et al., 2014)")
print(f"Parametros: epoch=100, pop_size=30, n_vars={N_FEATURES}")
print("Transfer fn: V-shape T(x)=|2/pi·arctan(pi/2·x)|  (Emary et al., 2016)")
print("─" * 68)

try:
    from mealpy import GWO, FloatVar, Problem

    # ── Definición del problema ────────────────────────────────────────────────
    class GWOFeatureSelection(Problem):
        def __init__(self, bounds, minmax, **kwargs):
            super().__init__(bounds, minmax, **kwargs)

        def obj_func(self, solution):
            # solution: vector continuo [-6, 6]^N
            # Transfer function V-shape → binario
            binary = (np.abs((2 / np.pi) * np.arctan((np.pi / 2) * solution)) > 0.5).astype(float)
            return fitness_fn_array(binary)

    bounds = FloatVar(lb=(-6,) * N_FEATURES, ub=(6,) * N_FEATURES, name="features")
    problem = GWOFeatureSelection(bounds=bounds, minmax="max", log_to=None, seed=SEED)

    EPOCH    = 100
    POP_SIZE = 30

    t0 = time.time()
    optimizer = GWO.OriginalGWO(epoch=EPOCH, pop_size=POP_SIZE)
    optimizer.solve(problem, seed=SEED)
    t_gwo = time.time() - t0

    best_sol_cont = optimizer.g_best.solution
    best_binary   = (np.abs((2 / np.pi) * np.arctan((np.pi / 2) * best_sol_cont)) > 0.5).astype(float)
    n_sel         = int(best_binary.sum())
    f1_gwo        = optimizer.g_best.target.fitness

    print(f"\nGWO completado en {t_gwo:.1f}s")
    print(f"Features seleccionadas: {n_sel} / {N_FEATURES}  ({100*n_sel/N_FEATURES:.1f}%)")
    print(f"F1-macro (inner CV 3-fold): {f1_gwo:.4f}")
    print(f"Delta vs baseline: {'+' if f1_gwo >= f1_base else ''}{f1_gwo - f1_base:.4f}")

    # Historial de convergencia
    history_best = optimizer.history.list_global_best_fit

except ImportError as e:
    print(f"Error importando mealpy: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN FINAL CON OUTER CV (5-fold) — estimación de generalización
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 68)
print("EVALUACION EXPLORATORIA — StratifiedKFold(5) sobre features GWO")
print("─" * 68)
print("Nota: la selección GWO se hizo antes de esta comparación externa.")
print("Por ello, este valor no sustituye una validación anidada completa.")

OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
mask_gwo = best_binary.astype(bool)

# Baseline: todas las features, outer CV
f1s_base_outer = []
for tr, te in OUTER_CV.split(X_full, y):
    X_tr, X_te = X_full[tr], X_full[te]
    y_tr, y_te = y[tr], y[te]
    smote = SMOTE(k_neighbors=2, random_state=SEED)
    X_r, y_r = smote.fit_resample(X_tr, y_tr)
    clf = ComplementNB(); clf.fit(X_r, y_r)
    f1s_base_outer.append(f1_score(y_te, clf.predict(X_te), average='macro', zero_division=0))

# GWO: features seleccionadas, outer CV
X_gwo = X_full[:, mask_gwo]
f1s_gwo_outer = []
for tr, te in OUTER_CV.split(X_gwo, y):
    X_tr, X_te = X_gwo[tr], X_gwo[te]
    y_tr, y_te = y[tr], y[te]
    smote = SMOTE(k_neighbors=2, random_state=SEED)
    X_r, y_r = smote.fit_resample(X_tr, y_tr)
    clf = ComplementNB(); clf.fit(X_r, y_r)
    f1s_gwo_outer.append(f1_score(y_te, clf.predict(X_te), average='macro', zero_division=0))

f1_base_out = np.mean(f1s_base_outer); std_base_out = np.std(f1s_base_outer)
f1_gwo_out  = np.mean(f1s_gwo_outer);  std_gwo_out  = np.std(f1s_gwo_outer)

print(f"  Baseline (400 feat)  : F1={f1_base_out:.4f} ± {std_base_out:.4f}  "
      f"folds={[round(x,3) for x in f1s_base_outer]}")
print(f"  GWO ({n_sel:>3} feat)    : F1={f1_gwo_out:.4f} ± {std_gwo_out:.4f}  "
      f"folds={[round(x,3) for x in f1s_gwo_outer]}")
print(f"  Delta F1             : {'+' if f1_gwo_out >= f1_base_out else ''}{f1_gwo_out - f1_base_out:.4f}")
print(f"  Reduccion de espacio : {100*(1 - n_sel/N_FEATURES):.1f}%  ({N_FEATURES} → {n_sel} features)")

# ── Top 30 features seleccionadas ────────────────────────────────────────────
selected_names = feature_names[mask_gwo]
# Puntaje TF-IDF medio para ordenar
tfidf_mean = X_full[:, mask_gwo].mean(axis=0)
top_idx    = np.argsort(tfidf_mean)[::-1][:30]
print(f"\n  Top 30 features seleccionadas (por TF-IDF medio):")
for i, idx in enumerate(top_idx):
    print(f"    {i+1:>2}. '{selected_names[idx]}'  (tfidf_mean={tfidf_mean[idx]:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
df_results = pd.DataFrame({
    'Modelo':   ['Baseline (400 feat)', f'GWO ({n_sel} feat)'],
    'n_features': [N_FEATURES, n_sel],
    'F1_media': [f1_base_out, f1_gwo_out],
    'F1_std':   [std_base_out, std_gwo_out],
    'IC95':     [1.96*std_base_out/np.sqrt(5), 1.96*std_gwo_out/np.sqrt(5)],
})
df_results.to_csv(os.path.join(OUT_DIR, 'gwo_resultados.csv'), index=False, encoding='utf-8-sig')

df_features = pd.DataFrame({
    'feature': selected_names,
    'tfidf_mean': X_full[:, mask_gwo].mean(axis=0),
    'tfidf_std':  X_full[:, mask_gwo].std(axis=0),
})
df_features = df_features.sort_values('tfidf_mean', ascending=False)
df_features.to_csv(os.path.join(OUT_DIR, 'gwo_features_seleccionadas.csv'),
                   index=False, encoding='utf-8-sig')
print(f"\nCSVs guardados: gwo_resultados.csv, gwo_features_seleccionadas.csv")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── 1. Curva de convergencia ──────────────────────────────────────────────────
ax = axes[0]
ax.plot(range(1, len(history_best) + 1), history_best,
        color='#1F497D', linewidth=2)
ax.axhline(f1_base, color='#888888', linestyle='--', linewidth=1.5,
           label=f'Baseline {f1_base:.3f}')
ax.set_xlabel('Epoch'); ax.set_ylabel('F1-macro (fitness)')
ax.set_title('GWO — Curva de convergencia\n(fitness del alfa por epoch)', fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── 2. Comparación barras (outer CV) ─────────────────────────────────────────
ax = axes[1]
labels = [f'Baseline\n(400 feat)', f'GWO\n({n_sel} feat)']
vals   = [f1_base_out, f1_gwo_out]
errs   = [1.96*std_base_out/np.sqrt(5), 1.96*std_gwo_out/np.sqrt(5)]
colors = ['#888888', '#1F497D']
bars   = ax.bar([0, 1], vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
ax.errorbar([0, 1], vals, yerr=errs, fmt='none', color='#222222', capsize=7, linewidth=2)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.015, f'{val:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('F1-macro (5-fold outer CV)'); ax.set_ylim(0, 1.05)
ax.set_title('F1-macro con IC 95%\n(Outer CV 5-fold)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# ── 3. Top 20 features seleccionadas ─────────────────────────────────────────
ax = axes[2]
top20 = df_features.head(20)
y_pos = np.arange(len(top20))
ax.barh(y_pos, top20['tfidf_mean'], color='#1E6823', alpha=0.8, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(top20['feature'], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('TF-IDF medio en corpus')
ax.set_title(f'Top 20 features seleccionadas por GWO\n(de {n_sel} totales seleccionadas)', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gwo_seleccion.png'), dpi=150)
plt.close()
print("Grafica guardada: gwo_seleccion.png")

# ── Mapa de features seleccionadas vs no seleccionadas ───────────────────────
fig, ax = plt.subplots(figsize=(14, 3))
colors_map = ['#D5E8F0' if b else '#F8F8F8' for b in best_binary]
for i, (c, b) in enumerate(zip(colors_map, best_binary)):
    ax.bar(i, 1, width=1, color='#1F497D' if b else '#EEEEEE',
           edgecolor='none', linewidth=0)
ax.set_xlim(0, N_FEATURES); ax.set_ylim(0, 1)
ax.set_xlabel('Indice de feature (0–399)')
ax.set_title(f'Mapa binario de seleccion GWO: {n_sel} features seleccionadas (azul) '
             f'/ {N_FEATURES - n_sel} descartadas (gris)', fontsize=10)
ax.set_yticks([])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#1F497D', label='Seleccionada'),
                   Patch(color='#EEEEEE', label='Descartada')],
          loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gwo_mapa_features.png'), dpi=150)
plt.close()
print("Grafica guardada: gwo_mapa_features.png")

# ── Tabla final ───────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("RESUMEN FINAL")
print("=" * 68)
print(f"  {'Metodo':<30} {'Features':>8}  {'F1-macro':>8}  {'±IC95%':>8}")
print(f"  {'-'*56}")
print(f"  {'Baseline (sin seleccion)':<30} {N_FEATURES:>8}  {f1_base_out:>8.4f}  {1.96*std_base_out/np.sqrt(5):>8.4f}")
print(f"  {'GWO (Mirjalili et al., 2014)':<30} {n_sel:>8}  {f1_gwo_out:>8.4f}  {1.96*std_gwo_out/np.sqrt(5):>8.4f}")
print(f"\n  Reduccion espacio de features: {N_FEATURES} → {n_sel} ({100*(1-n_sel/N_FEATURES):.1f}% menos)")
print(f"  Delta F1: {'+' if f1_gwo_out >= f1_base_out else ''}{f1_gwo_out - f1_base_out:.4f}")
print("\n" + "=" * 68)
print("FIN — gwo_feature_selection.py")
print("=" * 68)
