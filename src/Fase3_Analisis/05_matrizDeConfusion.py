import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_SALIDA_IMAGEN = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "matriz_confusion.png"))

def main():
    print("Generando Matriz de Confusión para el informe...")
    
    # 1. Cargar datos
    df = pd.read_csv(RUTA_DATOS, encoding='utf-8-sig')
    X = df['perfil_limpio'].values
    y = df['grado'].values
    
    # 2. Vectorización (Sincronizado con max_features=400)
    vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)
    
    # 3. Split (Stratified para respetar desbalance)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42, stratify=y)
    
    # 4. Clasificador (Random Forest como modelo base del reporte)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # 5. Generar Matriz
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    
    # 6. Plot y Guardado
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Matriz de Confusión del Clasificador', fontweight='bold')
    plt.tight_layout()
    plt.savefig(RUTA_SALIDA_IMAGEN, dpi=300)
    print(f"Matriz de confusión exportada exitosamente en: {RUTA_SALIDA_IMAGEN}")

if __name__ == "__main__":
    main()