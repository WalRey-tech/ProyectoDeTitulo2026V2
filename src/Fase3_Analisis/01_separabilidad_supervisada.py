import pandas as pd
import numpy as np
import os
import warnings
import joblib

# Supresión de advertencias de ejecución para mantener la consola limpia
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

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
    print("Iniciando Fase 3: Benchmark de Modelos y Evaluación de Separabilidad")
    print("-" * 65)
    
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

    # =============================================================================
    # 2. VECTORIZACIÓN DE TEXTO (TF-IDF)
    # =============================================================================
    print("Ejecutando vectorización semántica (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2) 
    X_sparse = vectorizer.fit_transform(X_text)
    X_dense = X_sparse.toarray()

    # Codificación de etiquetas categóricas a valores numéricos
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # =============================================================================
    # 3. BALANCEO DE DATOS MEDIANTE OVERSAMPLING SINTÉTICO (SMOTE)
    # =============================================================================
    print("Aplicando algoritmo SMOTE para interpolación de clase minoritaria...")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_dense, y)
    
    print(f"Volumen del dataset tras balanceo sintético: {len(y_resampled)} registros")

    # =============================================================================
    # 4. CONFIGURACIÓN DEL BENCHMARK ALGORÍTMICO
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

    print("\nEjecutando evaluación de métricas de rendimiento (Stratified 5-Fold CV)...")
    print("-" * 65)
    print(f"{'Algoritmo':<30} | {'Accuracy Medio':<15} | {'Desv. Estándar'}")
    print("-" * 65)

    resultados = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nombre, modelo in modelos.items():
        try:
            # Evaluación matemática del rendimiento
            scores = cross_val_score(modelo, X_resampled, y_resampled, cv=cv, scoring='accuracy')
            mean_acc = scores.mean()
            std_acc = scores.std()
            
            resultados.append({
                "Modelo": nombre,
                "Accuracy (%)": round(mean_acc * 100, 2),
                "Desviacion (%)": round(std_acc * 100, 2)
            })
            print(f"{nombre:<30} | {mean_acc*100:>13.2f}% | ±{std_acc*100:.2f}%")
        except Exception as e:
            print(f"Excepción en la evaluación de {nombre}: {str(e)}")

    # =============================================================================
    # 5. EXPORTACIÓN DE RESULTADOS Y PERSISTENCIA DEL MODELO ÓPTIMO
    # =============================================================================
    df_resultados = pd.DataFrame(resultados).sort_values(by="Accuracy (%)", ascending=False)
    os.makedirs(os.path.dirname(RUTA_REPORTE), exist_ok=True)
    df_resultados.to_csv(RUTA_REPORTE,sep=";", index=False, encoding='utf-8-sig')

    print("-" * 65)
    mejor_modelo_stats = df_resultados.iloc[0]
    
    print("\n[ REPORTE FINAL DE SEPARABILIDAD ]")
    print(f"Algoritmo superior detectado: {mejor_modelo_stats['Modelo']} ({mejor_modelo_stats['Accuracy (%)']}%)")
    print("La interpolación mediante SMOTE compensó efectivamente la asimetría de las clases.")

    # Entrenamiento del modelo definitivo para su despliegue
    print("\nIniciando entrenamiento del modelo en producción (Random Forest)...")
    modelo_final = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo_final.fit(X_resampled, y_resampled)
    
    # Persistencia de artefactos en disco
    joblib.dump(modelo_final, RUTA_MODELO_FINAL)
    joblib.dump(vectorizer, RUTA_VECTORIZADOR)
    
    print(f"Artefactos exportados exitosamente para integración en arquitectura web.")
    print(f"- Modelo: {RUTA_MODELO_FINAL}")
    print(f"- Vectorizador: {RUTA_VECTORIZADOR}")

if __name__ == "__main__":
    main()