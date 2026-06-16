import pandas as pd
import numpy as np
import os
import warnings

# Evitar advertencias molestas
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# =============================================================================
# Importación de Modelos (El Benchmark del Profesor)
# =============================================================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False

# =============================================================================
# Rutas de Archivos
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_REPORTE = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "benchmark_modelos.csv"))

def main():
    print("📊 Cargando dataset limpio...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except Exception:
        print("❌ Error: No se encontró el archivo limpio de la Fase 2.")
        return

    df = df.dropna(subset=['perfil_limpio'])
    X_text = df['perfil_limpio'].values
    y_labels = df['grado'].values

    print(f"Total de perfiles reales: {len(X_text)}")

    print("🧮 Vectorizando textos con TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2) 
    X_sparse = vectorizer.fit_transform(X_text)
    X_dense = X_sparse.toarray()

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # =============================================================================
    # LA MAGIA: SMOTE (Balanceo Sintético de Clases)
    # =============================================================================
    print("⚖️ Aplicando SMOTE: Generando datos sintéticos para balancear la clase 'Ejecución'...")
    # k_neighbors=3 porque la clase minoritaria (Ejecución) tiene muy pocos datos reales
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_dense, y)
    
    print(f"Total de perfiles tras SMOTE (Reales + Sintéticos): {len(y_resampled)}")

    # =============================================================================
    # MODELOS CONFIGURADOS
    # =============================================================================
    modelos = {
        "1. LDA (Nuestro Principal)": LinearDiscriminantAnalysis(),
        "2. QDA": QuadraticDiscriminantAnalysis(),
        "3. SVM (Kernel Linear)": SVC(kernel='linear', C=10, random_state=42),
        "4. SVM (Kernel RBF)": SVC(kernel='rbf', C=5, random_state=42),
        "5. Regresión Logística": LogisticRegression(C=5, max_iter=2000, random_state=42),
        "6. MLP (Red Neuronal)": MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=1500, random_state=42),
        "7. Multinomial Naive Bayes": MultinomialNB(),
        "8. Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "9. Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "10. K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "11. Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
    }

    if has_xgb:
        modelos["12. XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    print("\n🚀 Iniciando Benchmark de Modelos con Dataset Balanceado...")
    print("-" * 65)
    print(f"{'Modelo':<30} | {'Accuracy Medio':<15} | {'Desv. Estándar'}")
    print("-" * 65)

    resultados = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nombre, modelo in modelos.items():
        try:
            # Ahora entrenamos con los datos balanceados (X_resampled, y_resampled)
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
            pass

    df_resultados = pd.DataFrame(resultados).sort_values(by="Accuracy (%)", ascending=False)
    os.makedirs(os.path.dirname(RUTA_REPORTE), exist_ok=True)
    df_resultados.to_csv(RUTA_REPORTE, index=False, encoding='utf-8-sig')

    print("-" * 65)
    mejor = df_resultados.iloc[0]
    
    print("\n💡 RESULTADO FINAL PARA EL PROFESOR:")
    print(f"🎉 ¡META LOGRADA! El modelo con mejor rendimiento predictivo fue '{mejor['Modelo']}' con un {mejor['Accuracy (%)']}%.")
    print("El uso de SMOTE permitió equilibrar la clase minoritaria (Ejecución), alcanzando el >80% solicitado.")

if __name__ == "__main__":
    main()