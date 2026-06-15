import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_MATRIZ_LR = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "matriz_confusion_LR.png"))
RUTA_MATRIZ_SVC = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "matriz_confusion_SVC.png"))

def graficar_matriz(y_true, y_pred, clases, titulo, ruta_salida):
    """Función auxiliar para generar y guardar las matrices de confusión"""
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title(titulo)
    plt.ylabel('Grado Real')
    plt.xlabel('Grado Predicho por la IA')
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()

def main():
    print("Cargando dataset limpio para Análisis de Separabilidad...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: No se encontró {RUTA_ENTRADA}.")
        return

    df = df.dropna(subset=['perfil_limpio'])
    X = df['perfil_limpio']
    y = df['grado']
    clases = np.unique(y)

    # =============================================================================
    # 2. VECTORIZACIÓN Y ESCALADO (Instrucción del profesor: StandardScaler)
    # =============================================================================
    print("Vectorizando textos (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    print("Aplicando StandardScaler...")
    # with_mean=False es obligatorio porque TF-IDF genera una matriz dispersa (sparse matrix)
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_vec)

    # =============================================================================
    # 3. CONFIGURACIÓN DE MODELOS Y VALIDACIÓN CRUZADA
    # =============================================================================
    # StratifiedKFold (k=5, random_state=42) como pidió el profesor
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Definimos los 3 modelos: Baseline, LogisticRegression y SVC
    modelos = {
        "Baseline (Clase Mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        "SVC (Kernel RBF)": SVC(kernel='rbf', class_weight='balanced', random_state=42)
    }

    print("\n" + "="*60)
    print("INICIANDO VALIDACIÓN CRUZADA ESTRATIFICADA (5 Folds)")
    print("="*60)

    # =============================================================================
    # 4. EVALUACIÓN DE MODELOS
    # =============================================================================
    for nombre, modelo in modelos.items():
        # cross_validate para obtener métricas promedio de los 5 folds
        scores = cross_validate(modelo, X_scaled, y, cv=cv, scoring=['accuracy', 'f1_macro'])
        
        acc_mean = scores['test_accuracy'].mean()
        f1_macro_mean = scores['test_f1_macro'].mean()
        
        print(f"\n Modelo: {nombre}")
        print(f"   - Accuracy Media: {acc_mean:.4f}")
        print(f"   - F1-Macro Media: {f1_macro_mean:.4f}")

        # Si no es el baseline, generamos la matriz de confusión consolidada
        if nombre != "Baseline (Clase Mayoritaria)":
            print(f"   - Generando matriz de confusión para {nombre}...")
            # cross_val_predict junta las predicciones de los 5 folds para hacer una matriz total
            y_pred_cv = cross_val_predict(modelo, X_scaled, y, cv=cv)
            
            # Guardamos la matriz
            ruta = RUTA_MATRIZ_LR if nombre == "LogisticRegression" else RUTA_MATRIZ_SVC
            titulo = f'Matriz de Confusión - {nombre}\n(Validación Cruzada 5-Folds)'
            graficar_matriz(y, y_pred_cv, clases, titulo, ruta)
            
            # Imprimimos el reporte detallado para ver qué grados se confunden
            print("\n   Reporte de Clasificación Detallado:")
            print(classification_report(y, y_pred_cv))

    print("\n ¡Análisis 1 completado! Revisa las matrices generadas en la carpeta 'processed'.")

if __name__ == "__main__":
    main()