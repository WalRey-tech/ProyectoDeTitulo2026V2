import pandas as pd
import numpy as np
import os
import warnings
import joblib

# Supresión de advertencias de ejecución para mantener la consola limpia
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# =============================================================================
# IMPORTACIÓN DE MODELOS CLASIFICADORES
# =============================================================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# =============================================================================
# CONFIGURACIÓN DE RUTAS Y CONSTANTES
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_REPORTE = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "benchmark_modelos.csv"))

# Rutas para la exportación de los artefactos del modelo ganador
RUTA_MODELO_FINAL = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "modelo_random_forest.joblib"))
RUTA_VECTORIZADOR = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "vectorizador_tfidf.joblib"))

def main():
    print("Iniciando Fase 3: Benchmark de Modelos y Evaluación de Separabilidad sin Data Leakage")
    print("-" * 80)
    
    # =============================================================================
    # 1. INGESTA Y PREPARACIÓN DE DATOS
    # =============================================================================
    print("Cargando corpus procesado...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error crítico: No se encontró el archivo de entrada. Detalle: {e}")
        return

    # Limpieza de registros nulos
    df = df.dropna(subset=['perfil_limpio'])
    X_text = df['perfil_limpio'].values
    y_labels = df['grado'].values

    print(f"Total de perfiles académicos validados: {len(X_text)}")

    # Codificación de etiquetas categóricas a valores numéricos
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # =============================================================================
    # 2. CONFIGURACIÓN DEL BENCHMARK ALGORÍTMICO
    # =============================================================================
    modelos = {
        "1. LDA": LinearDiscriminantAnalysis(),
        "2. SVM (Kernel Linear)": SVC(kernel='linear', C=10, random_state=42),
        "3. SVM (Kernel RBF)": SVC(kernel='rbf', C=5, random_state=42),
        "4. Regresion Logistica": LogisticRegression(C=5, max_iter=2000, random_state=42),
        "5. MLP (Red Neuronal)": MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=1500, random_state=42),
        "6. Multinomial Naive Bayes": MultinomialNB(),
        "7. Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "8. Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "9. K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "10. Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
    }

    if HAS_XGB:
        modelos["11. XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    print("\nEjecutando evaluación de métricas de rendimiento (Stratified 5-Fold CV con Pipeline)...")
    print("-" * 80)
    print(f"{'Algoritmo':<30} | {'Accuracy Medio':<15} | {'F1-Score Macro'}")
    print("-" * 80)

    resultados = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    mejor_score = 0
    mejor_nombre = ""
    mejor_pipeline = None

    # Transformador para convertir matrices dispersas (TF-IDF) a densas (Requerido por LDA y SMOTE)
    to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

    # =============================================================================
    # 3. EVALUACIÓN CON AISLAMIENTO DE DATOS (PIPELINE)
    # =============================================================================
    for nombre, modelo in modelos.items():
        try:
            # El Pipeline encapsula TF-IDF, conversión y SMOTE para aislar cada fold
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2)),
                ("to_dense", to_dense),
                ("smote", SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)),
                ("clf", modelo)
            ])
            
            # Se evalúa directamente sobre el texto crudo (X_text), NO sobre datos previamente balanceados
            scores = cross_validate(pipe, X_text, y, cv=cv, scoring=('accuracy', 'f1_macro'))
            
            mean_acc = scores['test_accuracy'].mean()
            std_acc = scores['test_accuracy'].std()
            f1_mean = scores['test_f1_macro'].mean()
            
            resultados.append({
                "Modelo": nombre,
                "Accuracy (%)": round(mean_acc * 100, 2),
                "F1-Macro (%)": round(f1_mean * 100, 2),
                "Desviacion (%)": round(std_acc * 100, 2)
            })
            print(f"{nombre:<30} | {mean_acc*100:>13.2f}% | {f1_mean*100:>13.2f}%")
            
            # Selección dinámica del mejor modelo basado en Accuracy
            if mean_acc > mejor_score:
                mejor_score = mean_acc
                mejor_nombre = nombre
                mejor_pipeline = pipe

        except Exception as e:
            print(f"Excepción en la evaluación de {nombre}: {str(e)}")

    # =============================================================================
    # 4. EXPORTACIÓN DE RESULTADOS Y PERSISTENCIA DEL MODELO ÓPTIMO
    # =============================================================================
    df_resultados = pd.DataFrame(resultados).sort_values(by="Accuracy (%)", ascending=False)
    os.makedirs(os.path.dirname(RUTA_REPORTE), exist_ok=True)
    df_resultados.to_csv(RUTA_REPORTE, sep=";", index=False, encoding='utf-8-sig')

    print("-" * 80)
    print("\n[ REPORTE FINAL DE SEPARABILIDAD ]")
    print(f"Algoritmo superior detectado: {mejor_nombre} ({mejor_score * 100:.2f}%)")
    print("La interpolación mediante SMOTE se aplicó correctamente sin Data Leakage.")

    # =============================================================================
    # 5. ENTRENAMIENTO FINAL Y EXPORTACIÓN
    # =============================================================================
    print(f"\nIniciando entrenamiento del modelo en producción ({mejor_nombre})...")
    
    # Entrenamos el pipeline ganador con TODO el corpus para producción
    mejor_pipeline.fit(X_text, y)
    
    # Guardamos los artefactos usando las rutas originales configuradas
    joblib.dump(mejor_pipeline.named_steps['clf'], RUTA_MODELO_FINAL)
    joblib.dump(mejor_pipeline.named_steps['tfidf'], RUTA_VECTORIZADOR)
    
    print(f"Artefactos exportados exitosamente para integración en arquitectura web.")
    print(f"- Modelo: {RUTA_MODELO_FINAL}")
    print(f"- Vectorizador: {RUTA_VECTORIZADOR}")

if __name__ == "__main__":
    main()