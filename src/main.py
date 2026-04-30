import os
import pandas as pd
from config import SITES
from scraper import scrapear_sitio

RUTA_SALIDA = "data/raw/perfiles_egreso_raw.csv"

def limpiar_dataframe(df):
    # Limpieza secundaria por seguridad en Pandas
    df["perfil"] = df["perfil"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.fillna("")
    return df

def ordenar_columnas(df):
    columnas_ordenadas = [
        "universidad",
        "tipo_institucion",
        "carrera",
        "tipo_carrera",
        "url",
        "selector",
        "perfil",
        "error"
    ]
    columnas_presentes = [col for col in columnas_ordenadas if col in df.columns]
    return df[columnas_presentes]

def validar_dataframe(df):
    print("\n" + "="*40)
    print("📊 VALIDACIÓN PARA EL MODELO NLP")
    print("="*40)
    print(f"Total de instituciones procesadas: {len(df)}")
    
    perfiles_vacios = (df["perfil"].str.strip() == "").sum()
    
    # Subimos la exigencia: un perfil de egreso real tiene más de 150 caracteres.
    # Menos que eso, probablemente solo capturó un subtítulo y arruinará el análisis.
    perfiles_cortos = ((df["perfil"].str.len() > 0) & (df["perfil"].str.len() < 150)).sum()
    
    print(f"✅ Perfiles listos para análisis: {len(df) - perfiles_vacios - perfiles_cortos}")
    print(f"⚠️ Perfiles vacíos (Revisar selector CSS): {perfiles_vacios}")
    print(f"⚠️ Perfiles muy cortos (< 150 caracteres): {perfiles_cortos}")
    print("="*40)

def guardar_csv(df):
    # Usar os.path.dirname previene errores si la ruta cambia en el futuro
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    
    df.to_csv(
        RUTA_SALIDA,
        index=False,
        encoding="utf-8-sig",  # Vital para no romper los acentos en Excel
        sep=";"  
    )
    print(f"\n🚀 Archivo maestro guardado en: {RUTA_SALIDA}\n")

def main():
    resultados = []
    
    for site in SITES:
        try:
            print(f"\nScrapeando: {site['universidad']} - {site['carrera']}")
            data = scrapear_sitio(site)
            
            # 🔥 FEEDBACK EN VIVO: Te avisa exactamente qué pasó con este link
            if not data.get("perfil"):
                print("   ↳ Estado: ADVERTENCIA ⚠️ (Sin texto. Revisa el selector CSS en Chrome)")
            else:
                print("   ↳ Estado: OK ✅")
                
            resultados.append(data)
            
        except Exception as e:
            # Si el servidor de la U te bloqueó o la página no existe
            print(f"   ↳ Estado: ERROR ❌ ({str(e)})")
            resultados.append({
                "universidad": site.get("universidad", ""),
                "tipo_institucion": site.get("tipo_institucion", ""),
                "carrera": site.get("carrera", ""),
                "tipo_carrera": site.get("tipo_carrera", ""),
                "url": site.get("url", ""),
                "selector": site.get("selector", ""),
                "perfil": "",
                "error": str(e)
            })

    df = pd.DataFrame(resultados)

    # Procesamiento final
    df = limpiar_dataframe(df)
    df = ordenar_columnas(df)
    
    # Orden alfabético para facilitar la auditoría visual en Excel
    if "universidad" in df.columns:
        df = df.sort_values(by="universidad")
    
    validar_dataframe(df)
    guardar_csv(df)

if __name__ == "__main__":
    main()