# -*- coding: utf-8 -*-
"""
Cross-Validation 10-fold — Features seleccionadas por GWO
==========================================================
Modelo   : TF-IDF word (1,2)-grams + Complement NB
Features : subconjunto GWO cargado desde CSV (249 de 400 en la ejecución de referencia)
CV       : StratifiedKFold(10), SMOTE(k=2) dentro de cada fold
Métricas : F1-macro, accuracy, precision, recall por fold y por clase
"""

import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import ftfy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                              recall_score, confusion_matrix,
                              classification_report)
from sklearn.naive_bayes import ComplementNB
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# Carpeta actual:
# src/Fase3_Analisis/GWO/
BASE = os.path.dirname(os.path.abspath(__file__))

SRC_ROOT = os.path.abspath(
    os.path.join(BASE, "..")
)

# Corpus V2
CSV_V2 = os.path.join(
    SRC_ROOT,
    "data",
    "processed",
    "perfiles_egreso_etiquetado_v2.csv"
)

# Carpeta donde guardamos los resultados integrados
OUT_DIR = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "gwo"
)

# Features seleccionadas por GWO
GWO_CSV = os.path.join(
    OUT_DIR,
    "gwo_features_seleccionadas.csv"
)

os.makedirs(OUT_DIR, exist_ok=True)

SEED  = 42
np.random.seed(SEED)
CLASES = ['Civil', 'Ejecución', 'Informática']   # orden LabelEncoder

# ── Carga ──────────────────────────────────────────────────────────────────────
print("=" * 68)
print("CV 10-FOLD — TF-IDF + CNB con features seleccionadas por GWO")
print("=" * 68)

df = pd.read_csv(CSV_V2, encoding='utf-8-sig')
df['perfil_egreso'] = df['perfil_egreso'].apply(lambda x: ftfy.fix_text(str(x)))
df['grado']         = df['grado'].apply(lambda x: ftfy.fix_text(str(x)))
textos = df['perfil_egreso'].tolist()
le     = LabelEncoder()
y      = le.fit_transform(df['grado'].tolist())
clases_enc = list(le.classes_)          # ['Civil', 'Ejecución', 'Informática']
print(f"Dataset : {len(df)} docs — {dict(df['grado'].value_counts())}")

# ── Reconstruir máscara GWO desde CSV ─────────────────────────────────────────
df_gwo    = pd.read_csv(GWO_CSV, encoding='utf-8-sig')
gwo_feats = set(df_gwo['feature'].tolist())

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
X_full       = VEC.fit_transform(textos).toarray()
feature_names = np.array(VEC.get_feature_names_out())
mask_gwo     = np.array([f in gwo_feats for f in feature_names])
X_gwo        = X_full[:, mask_gwo]
n_feat       = X_gwo.shape[1]
print(f"Features GWO : {n_feat} (de {X_full.shape[1]} totales)")
if n_feat == 0:
    raise ValueError(
        "Ninguna característica del CSV coincide con el vocabulario TF-IDF. "
        "Compruebe que el corpus y gwo_features_seleccionadas.csv pertenecen "
        "a la misma ejecución."
    )
if n_feat != len(gwo_feats):
    print(
        f"ADVERTENCIA: el CSV contiene {len(gwo_feats)} características, pero "
        f"solo {n_feat} coinciden con el vocabulario reconstruido."
    )

# ══════════════════════════════════════════════════════════════════════════════
# CV 10-FOLD
# ══════════════════════════════════════════════════════════════════════════════
CV10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
CV5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=SEED)

def evaluar_cv(X, y, cv, label):
    resultados_fold = []
    y_true_all, y_pred_all = [], []

    print(f"\n{'─'*68}")
    print(f"{label}")
    print(f"{'─'*68}")
    print(f"  {'Fold':>4}  {'n_test':>6}  {'F1-macro':>9}  {'Accuracy':>9}  "
          f"{'F1-Civil':>9}  {'F1-Ejec':>8}  {'F1-Info':>8}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}")

    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # SMOTE dentro del fold
        smote = SMOTE(k_neighbors=2, random_state=SEED)
        try:
            X_r, y_r = smote.fit_resample(X_tr, y_tr)
        except ValueError:
            X_r, y_r = X_tr, y_tr     # si el fold es demasiado pequeño

        clf = ComplementNB()
        clf.fit(X_r, y_r)
        y_pred = clf.predict(X_te)

        f1_mac  = f1_score(y_te, y_pred, average='macro',    zero_division=0)
        acc     = accuracy_score(y_te, y_pred)
        f1_cls  = f1_score(y_te, y_pred, average=None,       zero_division=0,
                           labels=[0, 1, 2])

        print(f"  {fold:>4}  {len(y_te):>6}  {f1_mac:>9.4f}  {acc:>9.4f}  "
              f"{f1_cls[0]:>9.4f}  {f1_cls[1]:>8.4f}  {f1_cls[2]:>8.4f}")

        resultados_fold.append({
            'fold': fold, 'n_test': len(y_te),
            'F1_macro': f1_mac, 'Accuracy': acc,
            'F1_Civil': f1_cls[0], 'F1_Ejecucion': f1_cls[1], 'F1_Informatica': f1_cls[2],
        })
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)

    df_r  = pd.DataFrame(resultados_fold)
    y_ta  = np.array(y_true_all)
    y_pa  = np.array(y_pred_all)

    print(f"  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}")
    print(f"  {'Media':>4}  {'':>6}  {df_r['F1_macro'].mean():>9.4f}  "
          f"{df_r['Accuracy'].mean():>9.4f}  "
          f"{df_r['F1_Civil'].mean():>9.4f}  "
          f"{df_r['F1_Ejecucion'].mean():>8.4f}  "
          f"{df_r['F1_Informatica'].mean():>8.4f}")
    print(f"  {'Std':>4}  {'':>6}  {df_r['F1_macro'].std():>9.4f}  "
          f"{df_r['Accuracy'].std():>9.4f}  "
          f"{df_r['F1_Civil'].std():>9.4f}  "
          f"{df_r['F1_Ejecucion'].std():>8.4f}  "
          f"{df_r['F1_Informatica'].std():>8.4f}")

    n = len(df_r)
    ic95 = 1.96 * df_r['F1_macro'].std() / np.sqrt(n)
    print(f"\n  IC 95% F1-macro : ±{ic95:.4f}  "
          f"→  [{df_r['F1_macro'].mean()-ic95:.4f},  {df_r['F1_macro'].mean()+ic95:.4f}]")

    print(f"\n  Reporte agregado (predicciones concatenadas de todos los folds):")
    print(classification_report(y_ta, y_pa, target_names=clases_enc, zero_division=0))

    return df_r, y_ta, y_pa

# ── 5-fold (baseline de comparación) ─────────────────────────────────────────
df_5,  yt5,  yp5  = evaluar_cv(X_gwo, y, CV5,  "5-FOLD CV  — TF-IDF + CNB (features GWO)")

# ── 10-fold (objetivo principal) ──────────────────────────────────────────────
df_10, yt10, yp10 = evaluar_cv(X_gwo, y, CV10, "10-FOLD CV — TF-IDF + CNB (features GWO)")

# ── Guardar CSVs ──────────────────────────────────────────────────────────────
df_5.to_csv( os.path.join(OUT_DIR, 'cv5_gwo_resultados.csv'),  index=False, encoding='utf-8-sig')
df_10.to_csv(os.path.join(OUT_DIR, 'cv10_gwo_resultados.csv'), index=False, encoding='utf-8-sig')
print("\nCSVs guardados: cv5_gwo_resultados.csv, cv10_gwo_resultados.csv")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── 1. F1-macro por fold — 5 vs 10 ───────────────────────────────────────────
ax = axes[0]
ax.plot(df_5['fold'],  df_5['F1_macro'],  marker='o', linewidth=2,
        color='#888888', label='5-fold')
ax.plot(df_10['fold'], df_10['F1_macro'], marker='s', linewidth=2,
        color='#1F497D', label='10-fold')
ax.axhline(df_5['F1_macro'].mean(),  color='#888888', linestyle='--',
           linewidth=1, alpha=0.7)
ax.axhline(df_10['F1_macro'].mean(), color='#1F497D', linestyle='--',
           linewidth=1, alpha=0.7)
ax.fill_between(df_10['fold'],
                df_10['F1_macro'].mean() - 1.96*df_10['F1_macro'].std()/np.sqrt(10),
                df_10['F1_macro'].mean() + 1.96*df_10['F1_macro'].std()/np.sqrt(10),
                alpha=0.1, color='#1F497D')
ax.set_xlabel('Fold'); ax.set_ylabel('F1-macro')
ax.set_title('F1-macro por fold\n(features GWO, CNB)', fontsize=10)
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
ax.set_xticks(range(1, 11))

# ── 2. Boxplot comparativo 5 vs 10 fold ──────────────────────────────────────
ax = axes[1]
data_bp  = [df_5['F1_macro'].values, df_10['F1_macro'].values]
bp = ax.boxplot(data_bp, tick_labels=['5-fold', '10-fold'], patch_artist=True,
                medianprops=dict(color='#C00000', linewidth=2.5),
                widths=0.5)
colores_bp = ['#888888', '#1F497D']
for patch, c in zip(bp['boxes'], colores_bp):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for element in ['whiskers', 'caps']:
    for item in bp[element]: item.set_color('#333333')

# Puntos individuales superpuestos
for i, (vals, jitter_x) in enumerate([(df_5['F1_macro'], 1), (df_10['F1_macro'], 2)], 0):
    jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(np.full(len(vals), jitter_x) + jitter, vals,
               color=colores_bp[i], alpha=0.8, s=40, zorder=5)

ax.set_ylabel('F1-macro'); ax.set_ylim(0, 1.05)
ax.set_title('Distribución F1-macro\n5-fold vs 10-fold (GWO features)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

m5, s5   = df_5['F1_macro'].mean(),  df_5['F1_macro'].std()
m10, s10 = df_10['F1_macro'].mean(), df_10['F1_macro'].std()
ax.text(1, 0.03, f'μ={m5:.3f}\nσ={s5:.3f}',  ha='center', fontsize=9, color='#444444')
ax.text(2, 0.03, f'μ={m10:.3f}\nσ={s10:.3f}', ha='center', fontsize=9, color='#1F497D')

# ── 3. Matriz de confusión (10-fold agregada) ─────────────────────────────────
ax = axes[2]
cm = confusion_matrix(yt10, yp10)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=clases_enc, yticklabels=clases_enc,
            ax=ax, cbar=False, linewidths=0.5)
ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
ax.set_title('Matriz de confusión\n10-fold agregada (GWO features)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cv10_gwo_resultados.png'), dpi=150)
plt.close()
print("Gráfica guardada: cv10_gwo_resultados.png")

# ── F1 por clase por fold (10-fold) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
x = df_10['fold']
ax.plot(x, df_10['F1_Civil'],      marker='o', linewidth=2,
        color='#1F497D', label='Civil')
ax.plot(x, df_10['F1_Ejecucion'],  marker='s', linewidth=2,
        color='#BF5A00', label='Ejecución')
ax.plot(x, df_10['F1_Informatica'],marker='^', linewidth=2,
        color='#1E6823', label='Informática')
ax.axhline(df_10['F1_macro'].mean(), color='#C00000', linestyle='--',
           linewidth=1.5, label=f"F1-macro medio={df_10['F1_macro'].mean():.3f}")
ax.set_xlabel('Fold'); ax.set_ylabel('F1 por clase')
ax.set_title('F1 por clase en cada fold — 10-fold CV (GWO features, CNB)', fontsize=11)
ax.legend(loc='lower left'); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cv10_gwo_f1_clase.png'), dpi=150)
plt.close()
print("Gráfica guardada: cv10_gwo_f1_clase.png")

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN COMPARATIVO FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("RESUMEN COMPARATIVO — mismas features GWO")
print("=" * 68)
ic5  = 1.96 * s5  / np.sqrt(5)
ic10 = 1.96 * s10 / np.sqrt(10)
print(f"  {'CV':>6}  {'F1-macro':>9}  {'Std':>7}  {'IC 95%':>9}  {'IC inferior':>12}  {'IC superior':>12}")
print(f"  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*9}  {'─'*12}  {'─'*12}")
print(f"  {'5-fold':>6}  {m5:>9.4f}  {s5:>7.4f}  {ic5:>9.4f}  "
      f"{m5-ic5:>12.4f}  {m5+ic5:>12.4f}")
print(f"  {'10-fold':>6}  {m10:>9.4f}  {s10:>7.4f}  {ic10:>9.4f}  "
      f"{m10-ic10:>12.4f}  {m10+ic10:>12.4f}")
print(f"\n  Mejora 10-fold vs 5-fold: {'+' if m10>=m5 else ''}{m10-m5:.4f}  |  "
      f"Reduccion IC95%: {(ic5-ic10)/ic5*100:.1f}%")
print("=" * 68)
